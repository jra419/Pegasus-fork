#include <core.p4>
#include <tna.p4>

#include "includes/headers.p4"
#include "includes/util.p4"
#include "includes/parser.p4"
#include "includes/deparser.p4"
#include "includes/constants.p4"

control SwitchIngress(inout header_t hdr,
					  inout metadata_t ig_md,
					  in ingress_intrinsic_metadata_t ig_intr_md,
					  in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
					  inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
					  inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {
	action noaction() {}

	CRCPolynomial<bit<32>>(32w0x04C11DB7, true, false, false, 32w0xFFFFFFFF,
						   32w0xFFFFFFFF) crc32_poly;
	Hash<bit<32>>(HashAlgorithm_t.CUSTOM, crc32_poly) flow_hash;

	Register<bit<16>, bit<16>>(FLOW_TABLE_SIZE) flow_sig;
	Register<bit<32>, bit<16>>(FLOW_TABLE_SIZE) last_ts;
	Register<bit<16>, bit<16>>(FLOW_TABLE_SIZE) pkt_count;
	Register<bit<32>, bit<16>>(FLOW_TABLE_SIZE) byte_count;

	// Stores this packet's signature (hash[31:16])
	// Return 1 if the slot held another flow.
	// Registers are indexed by hash[15:0].
	RegisterAction<bit<16>, bit<16>, bit<1>>(flow_sig) swap_flow_sig = {
		void apply(inout bit<16> value, out bit<1> other_flow) {
			if (value != ig_md.flow.hash[31:16]) {
				other_flow = 1;
			} else {
				other_flow = 0;
			}
			value = ig_md.flow.hash[31:16];
		}
	};

	// Stores the current timestamp (2^8 ns units).
	// Return the time since the slot's previous packet.
	RegisterAction<bit<32>, bit<16>, bit<32>>(last_ts) swap_last_ts = {
		void apply(inout bit<32> value, out bit<32> gap) {
			gap = ig_intr_md.ingress_mac_tstamp[39:8] - value;
			value = ig_intr_md.ingress_mac_tstamp[39:8];
		}
	};

	// New flow: the count restarts at 1 (current packet is index 0).
	RegisterAction<bit<16>, bit<16>, bit<16>>(pkt_count) restart_pkt_count = {
		void apply(inout bit<16> value) {
			value = 1;
		}
	};

	// Same flow: return the count as this packet's index, then increments it (saturating).
	RegisterAction<bit<16>, bit<16>, bit<16>>(pkt_count) next_pkt_count = {
		void apply(inout bit<16> value, out bit<16> index) {
			index = value;
			value = value |+| 1;
		}
	};

	// New flow: the byte count restarts at this packet's length.
	RegisterAction<bit<32>, bit<16>, bit<32>>(byte_count) restart_byte_count = {
		void apply(inout bit<32> value) {
			value = (bit<32>)hdr.ipv4.total_len;
		}
	};

	// Same flow: adds this packet's length (saturating) and returns the new total.
	RegisterAction<bit<32>, bit<16>, bit<32>>(byte_count) next_byte_count = {
		void apply(inout bit<32> value, out bit<32> total) {
			value = value |+| (bit<32>)hdr.ipv4.total_len;
			total = value;
		}
	};

	action compute_flow_hash() {
		ig_md.flow.hash = flow_hash.get({hdr.ipv4.src_addr,
										 hdr.ipv4.dst_addr,
										 hdr.ipv4.protocol,
										 ig_md.flow.port_src,
										 ig_md.flow.port_dst});
	}

	// ---------------------------------------------------------------------------
	// Confidence: the top logit beats the second by at least CONF_THRESHOLD exactly when
	// some class i beats every other class j by that much, i.e. logit_i - (logit_j + threshold)
	// >= 0 for all j != i.
	// ---------------------------------------------------------------------------

	action add_conf_threshold() {
		ig_md.conf.t0 = hdr.output.logits.class_0 + CONF_THRESHOLD;
		ig_md.conf.t1 = hdr.output.logits.class_1 + CONF_THRESHOLD;
		ig_md.conf.t2 = hdr.output.logits.class_2 + CONF_THRESHOLD;
	}

	action compute_logit_margins() {
		ig_md.conf.d01 = hdr.output.logits.class_0 - ig_md.conf.t1;
		ig_md.conf.d02 = hdr.output.logits.class_0 - ig_md.conf.t2;
		ig_md.conf.d10 = hdr.output.logits.class_1 - ig_md.conf.t0;
		ig_md.conf.d12 = hdr.output.logits.class_1 - ig_md.conf.t2;
		ig_md.conf.d20 = hdr.output.logits.class_2 - ig_md.conf.t0;
		ig_md.conf.d21 = hdr.output.logits.class_2 - ig_md.conf.t1;
	}

	action set_confident() {
		hdr.output.confident = 1;
	}

	action set_not_confident() {
		hdr.output.confident = 0;
	}

	// Confident if both differences of some class are >= 0 (sign bit 0).
	table confidence_check {
		key = {
			ig_md.conf.d01[31:31] : ternary;
			ig_md.conf.d02[31:31] : ternary;
			ig_md.conf.d10[31:31] : ternary;
			ig_md.conf.d12[31:31] : ternary;
			ig_md.conf.d20[31:31] : ternary;
			ig_md.conf.d21[31:31] : ternary;
		}
		actions = {
			set_confident;
			set_not_confident;
		}
		const entries = {
			(0, 0, _, _, _, _) : set_confident();
			(_, _, 0, 0, _, _) : set_confident();
			(_, _, _, _, 0, 0) : set_confident();
		}
		size			= 4;
		default_action	= set_not_confident();
	}

	// ---------------------------------------------------------------------------
	// MM1: features -> hidden1. Each table's action carries its leaves' LUT rows for all
	// of hidden1 (16 packed words). The first two tables write separate accumulators so
	// they can run in parallel; mm1_flow_len adds into the first.
	// ---------------------------------------------------------------------------

	// Writes hidden1.
	action set_mm1_proto_ts_rel(bit<32> w0, bit<32> w1, bit<32> w2, bit<32> w3, bit<32> w4,
	                            bit<32> w5, bit<32> w6, bit<32> w7, bit<32> w8, bit<32> w9,
	                            bit<32> w10, bit<32> w11, bit<32> w12, bit<32> w13, bit<32> w14,
	                            bit<32> w15) {
		ig_md.hidden1.w0 = w0;
		ig_md.hidden1.w1 = w1;
		ig_md.hidden1.w2 = w2;
		ig_md.hidden1.w3 = w3;
		ig_md.hidden1.w4 = w4;
		ig_md.hidden1.w5 = w5;
		ig_md.hidden1.w6 = w6;
		ig_md.hidden1.w7 = w7;
		ig_md.hidden1.w8 = w8;
		ig_md.hidden1.w9 = w9;
		ig_md.hidden1.w10 = w10;
		ig_md.hidden1.w11 = w11;
		ig_md.hidden1.w12 = w12;
		ig_md.hidden1.w13 = w13;
		ig_md.hidden1.w14 = w14;
		ig_md.hidden1.w15 = w15;
	}

	// Writes hidden1_partial.
	action set_mm1_flags_pkt_idx(bit<32> w0, bit<32> w1, bit<32> w2, bit<32> w3, bit<32> w4,
	                             bit<32> w5, bit<32> w6, bit<32> w7, bit<32> w8, bit<32> w9,
	                             bit<32> w10, bit<32> w11, bit<32> w12, bit<32> w13, bit<32> w14,
	                             bit<32> w15) {
		ig_md.hidden1_partial.w0 = w0;
		ig_md.hidden1_partial.w1 = w1;
		ig_md.hidden1_partial.w2 = w2;
		ig_md.hidden1_partial.w3 = w3;
		ig_md.hidden1_partial.w4 = w4;
		ig_md.hidden1_partial.w5 = w5;
		ig_md.hidden1_partial.w6 = w6;
		ig_md.hidden1_partial.w7 = w7;
		ig_md.hidden1_partial.w8 = w8;
		ig_md.hidden1_partial.w9 = w9;
		ig_md.hidden1_partial.w10 = w10;
		ig_md.hidden1_partial.w11 = w11;
		ig_md.hidden1_partial.w12 = w12;
		ig_md.hidden1_partial.w13 = w13;
		ig_md.hidden1_partial.w14 = w14;
		ig_md.hidden1_partial.w15 = w15;
	}

	// Adds into hidden1.
	action set_mm1_flow_len(bit<32> w0, bit<32> w1, bit<32> w2, bit<32> w3, bit<32> w4, bit<32> w5,
	                        bit<32> w6, bit<32> w7, bit<32> w8, bit<32> w9, bit<32> w10,
	                        bit<32> w11, bit<32> w12, bit<32> w13, bit<32> w14, bit<32> w15) {
		ig_md.hidden1.w0 = w0 + ig_md.hidden1.w0;
		ig_md.hidden1.w1 = w1 + ig_md.hidden1.w1;
		ig_md.hidden1.w2 = w2 + ig_md.hidden1.w2;
		ig_md.hidden1.w3 = w3 + ig_md.hidden1.w3;
		ig_md.hidden1.w4 = w4 + ig_md.hidden1.w4;
		ig_md.hidden1.w5 = w5 + ig_md.hidden1.w5;
		ig_md.hidden1.w6 = w6 + ig_md.hidden1.w6;
		ig_md.hidden1.w7 = w7 + ig_md.hidden1.w7;
		ig_md.hidden1.w8 = w8 + ig_md.hidden1.w8;
		ig_md.hidden1.w9 = w9 + ig_md.hidden1.w9;
		ig_md.hidden1.w10 = w10 + ig_md.hidden1.w10;
		ig_md.hidden1.w11 = w11 + ig_md.hidden1.w11;
		ig_md.hidden1.w12 = w12 + ig_md.hidden1.w12;
		ig_md.hidden1.w13 = w13 + ig_md.hidden1.w13;
		ig_md.hidden1.w14 = w14 + ig_md.hidden1.w14;
		ig_md.hidden1.w15 = w15 + ig_md.hidden1.w15;
	}

	table mm1_proto_ts_rel {
		key = {
			hdr.ipv4.protocol : ternary;
			ig_md.feat.ts_rel : ternary;
		}
		actions = {
			set_mm1_proto_ts_rel;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm1_flags_pkt_idx {
		key = {
			ig_md.feat.tcp_flags    : ternary;
			ig_md.feat.packet_index : ternary;
		}
		actions = {
			set_mm1_flags_pkt_idx;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm1_flow_len {
		key = {
			ig_md.feat.flow_len : ternary;
		}
		actions = {
			set_mm1_flow_len;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	// ---------------------------------------------------------------------------
	// MM2: hidden1 -> hidden2. Codeblock c is hidden1 word c (two values). Codeblock 0
	// maps straight to LUT rows (mm2_direct). Codeblocks 1..15 only record their leaf
	// index (mm2_leaf_c), and the leaf indices are turned into LUT rows three codeblocks
	// at a time by exact-match tables (mm2_group_*).
	// ---------------------------------------------------------------------------

	// Writes hidden2.
	action set_mm2_direct(bit<32> w0, bit<32> w1, bit<32> w2, bit<32> w3, bit<32> w4, bit<32> w5,
	                      bit<32> w6, bit<32> w7) {
		ig_md.hidden2.w0 = w0;
		ig_md.hidden2.w1 = w1;
		ig_md.hidden2.w2 = w2;
		ig_md.hidden2.w3 = w3;
		ig_md.hidden2.w4 = w4;
		ig_md.hidden2.w5 = w5;
		ig_md.hidden2.w6 = w6;
		ig_md.hidden2.w7 = w7;
	}

	action set_mm2_leaf_1(bit<4> leaf) {
		ig_md.mm2_leaf.cb1 = leaf;
	}

	action set_mm2_leaf_2(bit<4> leaf) {
		ig_md.mm2_leaf.cb2 = leaf;
	}

	action set_mm2_leaf_3(bit<4> leaf) {
		ig_md.mm2_leaf.cb3 = leaf;
	}

	action set_mm2_leaf_4(bit<4> leaf) {
		ig_md.mm2_leaf.cb4 = leaf;
	}

	action set_mm2_leaf_5(bit<4> leaf) {
		ig_md.mm2_leaf.cb5 = leaf;
	}

	action set_mm2_leaf_6(bit<4> leaf) {
		ig_md.mm2_leaf.cb6 = leaf;
	}

	action set_mm2_leaf_7(bit<4> leaf) {
		ig_md.mm2_leaf.cb7 = leaf;
	}

	action set_mm2_leaf_8(bit<4> leaf) {
		ig_md.mm2_leaf.cb8 = leaf;
	}

	action set_mm2_leaf_9(bit<4> leaf) {
		ig_md.mm2_leaf.cb9 = leaf;
	}

	action set_mm2_leaf_10(bit<4> leaf) {
		ig_md.mm2_leaf.cb10 = leaf;
	}

	action set_mm2_leaf_11(bit<4> leaf) {
		ig_md.mm2_leaf.cb11 = leaf;
	}

	action set_mm2_leaf_12(bit<4> leaf) {
		ig_md.mm2_leaf.cb12 = leaf;
	}

	action set_mm2_leaf_13(bit<4> leaf) {
		ig_md.mm2_leaf.cb13 = leaf;
	}

	action set_mm2_leaf_14(bit<4> leaf) {
		ig_md.mm2_leaf.cb14 = leaf;
	}

	action set_mm2_leaf_15(bit<4> leaf) {
		ig_md.mm2_leaf.cb15 = leaf;
	}

	table mm2_direct {
		key = {
			ig_md.hidden1.w0 : ternary;
		}
		actions = {
			set_mm2_direct;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_1 {
		key = {
			ig_md.hidden1.w1 : ternary;
		}
		actions = {
			set_mm2_leaf_1;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_2 {
		key = {
			ig_md.hidden1.w2 : ternary;
		}
		actions = {
			set_mm2_leaf_2;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_3 {
		key = {
			ig_md.hidden1.w3 : ternary;
		}
		actions = {
			set_mm2_leaf_3;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_4 {
		key = {
			ig_md.hidden1.w4 : ternary;
		}
		actions = {
			set_mm2_leaf_4;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_5 {
		key = {
			ig_md.hidden1.w5 : ternary;
		}
		actions = {
			set_mm2_leaf_5;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_6 {
		key = {
			ig_md.hidden1.w6 : ternary;
		}
		actions = {
			set_mm2_leaf_6;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_7 {
		key = {
			ig_md.hidden1.w7 : ternary;
		}
		actions = {
			set_mm2_leaf_7;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_8 {
		key = {
			ig_md.hidden1.w8 : ternary;
		}
		actions = {
			set_mm2_leaf_8;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_9 {
		key = {
			ig_md.hidden1.w9 : ternary;
		}
		actions = {
			set_mm2_leaf_9;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_10 {
		key = {
			ig_md.hidden1.w10 : ternary;
		}
		actions = {
			set_mm2_leaf_10;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_11 {
		key = {
			ig_md.hidden1.w11 : ternary;
		}
		actions = {
			set_mm2_leaf_11;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_12 {
		key = {
			ig_md.hidden1.w12 : ternary;
		}
		actions = {
			set_mm2_leaf_12;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_13 {
		key = {
			ig_md.hidden1.w13 : ternary;
		}
		actions = {
			set_mm2_leaf_13;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_14 {
		key = {
			ig_md.hidden1.w14 : ternary;
		}
		actions = {
			set_mm2_leaf_14;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_leaf_15 {
		key = {
			ig_md.hidden1.w15 : ternary;
		}
		actions = {
			set_mm2_leaf_15;
			noaction;
		}
		size			= 2048;
		default_action	= noaction();
		idle_timeout	= false;
	}

	// Adds into hidden2.
	action set_mm2_group_1_3(bit<32> w0, bit<32> w1, bit<32> w2, bit<32> w3, bit<32> w4,
	                         bit<32> w5, bit<32> w6, bit<32> w7) {
		ig_md.hidden2.w0 = w0 + ig_md.hidden2.w0;
		ig_md.hidden2.w1 = w1 + ig_md.hidden2.w1;
		ig_md.hidden2.w2 = w2 + ig_md.hidden2.w2;
		ig_md.hidden2.w3 = w3 + ig_md.hidden2.w3;
		ig_md.hidden2.w4 = w4 + ig_md.hidden2.w4;
		ig_md.hidden2.w5 = w5 + ig_md.hidden2.w5;
		ig_md.hidden2.w6 = w6 + ig_md.hidden2.w6;
		ig_md.hidden2.w7 = w7 + ig_md.hidden2.w7;
	}

	// Writes hidden2_partial.
	action set_mm2_group_4_6(bit<32> w0, bit<32> w1, bit<32> w2, bit<32> w3, bit<32> w4,
	                         bit<32> w5, bit<32> w6, bit<32> w7) {
		ig_md.hidden2_partial.w0 = w0;
		ig_md.hidden2_partial.w1 = w1;
		ig_md.hidden2_partial.w2 = w2;
		ig_md.hidden2_partial.w3 = w3;
		ig_md.hidden2_partial.w4 = w4;
		ig_md.hidden2_partial.w5 = w5;
		ig_md.hidden2_partial.w6 = w6;
		ig_md.hidden2_partial.w7 = w7;
	}

	// Adds into hidden2.
	action set_mm2_group_7_9(bit<32> w0, bit<32> w1, bit<32> w2, bit<32> w3, bit<32> w4,
	                         bit<32> w5, bit<32> w6, bit<32> w7) {
		ig_md.hidden2.w0 = w0 + ig_md.hidden2.w0;
		ig_md.hidden2.w1 = w1 + ig_md.hidden2.w1;
		ig_md.hidden2.w2 = w2 + ig_md.hidden2.w2;
		ig_md.hidden2.w3 = w3 + ig_md.hidden2.w3;
		ig_md.hidden2.w4 = w4 + ig_md.hidden2.w4;
		ig_md.hidden2.w5 = w5 + ig_md.hidden2.w5;
		ig_md.hidden2.w6 = w6 + ig_md.hidden2.w6;
		ig_md.hidden2.w7 = w7 + ig_md.hidden2.w7;
	}

	// Adds into hidden2_partial.
	action set_mm2_group_10_12(bit<32> w0, bit<32> w1, bit<32> w2, bit<32> w3, bit<32> w4,
	                           bit<32> w5, bit<32> w6, bit<32> w7) {
		ig_md.hidden2_partial.w0 = w0 + ig_md.hidden2_partial.w0;
		ig_md.hidden2_partial.w1 = w1 + ig_md.hidden2_partial.w1;
		ig_md.hidden2_partial.w2 = w2 + ig_md.hidden2_partial.w2;
		ig_md.hidden2_partial.w3 = w3 + ig_md.hidden2_partial.w3;
		ig_md.hidden2_partial.w4 = w4 + ig_md.hidden2_partial.w4;
		ig_md.hidden2_partial.w5 = w5 + ig_md.hidden2_partial.w5;
		ig_md.hidden2_partial.w6 = w6 + ig_md.hidden2_partial.w6;
		ig_md.hidden2_partial.w7 = w7 + ig_md.hidden2_partial.w7;
	}

	// Adds into hidden2.
	action set_mm2_group_13_15(bit<32> w0, bit<32> w1, bit<32> w2, bit<32> w3, bit<32> w4,
	                           bit<32> w5, bit<32> w6, bit<32> w7) {
		ig_md.hidden2.w0 = w0 + ig_md.hidden2.w0;
		ig_md.hidden2.w1 = w1 + ig_md.hidden2.w1;
		ig_md.hidden2.w2 = w2 + ig_md.hidden2.w2;
		ig_md.hidden2.w3 = w3 + ig_md.hidden2.w3;
		ig_md.hidden2.w4 = w4 + ig_md.hidden2.w4;
		ig_md.hidden2.w5 = w5 + ig_md.hidden2.w5;
		ig_md.hidden2.w6 = w6 + ig_md.hidden2.w6;
		ig_md.hidden2.w7 = w7 + ig_md.hidden2.w7;
	}

	table mm2_group_1_3 {
		key = {
			ig_md.mm2_leaf.cb1 : exact;
			ig_md.mm2_leaf.cb2 : exact;
			ig_md.mm2_leaf.cb3 : exact;
		}
		actions = {
			set_mm2_group_1_3;
			noaction;
		}
		size			= 8192;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_group_4_6 {
		key = {
			ig_md.mm2_leaf.cb4 : exact;
			ig_md.mm2_leaf.cb5 : exact;
			ig_md.mm2_leaf.cb6 : exact;
		}
		actions = {
			set_mm2_group_4_6;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_group_7_9 {
		key = {
			ig_md.mm2_leaf.cb7 : exact;
			ig_md.mm2_leaf.cb8 : exact;
			ig_md.mm2_leaf.cb9 : exact;
		}
		actions = {
			set_mm2_group_7_9;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_group_10_12 {
		key = {
			ig_md.mm2_leaf.cb10 : exact;
			ig_md.mm2_leaf.cb11 : exact;
			ig_md.mm2_leaf.cb12 : exact;
		}
		actions = {
			set_mm2_group_10_12;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm2_group_13_15 {
		key = {
			ig_md.mm2_leaf.cb13 : exact;
			ig_md.mm2_leaf.cb14 : exact;
			ig_md.mm2_leaf.cb15 : exact;
		}
		actions = {
			set_mm2_group_13_15;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	// ---------------------------------------------------------------------------
	// MM3: hidden2 -> logits. mm3_word_k keys on hidden2 word k (two codeblocks) and
	// carries unpacked per-class logits. Even words accumulate into logits_partial, odd
	// words into the output header; the two are summed at the end.
	// ---------------------------------------------------------------------------

	action set_mm3_word_0(bit<32> class_0, bit<32> class_1, bit<32> class_2, bit<32> class_3) {
		ig_md.logits_partial.class_0 = class_0;
		ig_md.logits_partial.class_1 = class_1;
		ig_md.logits_partial.class_2 = class_2;
		//ig_md.logits_partial.class_3 = class_3;
	}

	action set_mm3_word_1(bit<32> class_0, bit<32> class_1, bit<32> class_2, bit<32> class_3) {
		hdr.output.logits.class_0 = class_0;
		hdr.output.logits.class_1 = class_1;
		hdr.output.logits.class_2 = class_2;
		//hdr.output.logits.class_3 = class_3;
	}

	action set_mm3_word_2(bit<32> class_0, bit<32> class_1, bit<32> class_2, bit<32> class_3) {
		ig_md.logits_partial.class_0 = class_0 + ig_md.logits_partial.class_0;
		ig_md.logits_partial.class_1 = class_1 + ig_md.logits_partial.class_1;
		ig_md.logits_partial.class_2 = class_2 + ig_md.logits_partial.class_2;
		//ig_md.logits_partial.class_3 = class_3 + ig_md.logits_partial.class_3;
	}

	action set_mm3_word_3(bit<32> class_0, bit<32> class_1, bit<32> class_2, bit<32> class_3) {
		hdr.output.logits.class_0 = class_0 + hdr.output.logits.class_0;
		hdr.output.logits.class_1 = class_1 + hdr.output.logits.class_1;
		hdr.output.logits.class_2 = class_2 + hdr.output.logits.class_2;
		//hdr.output.logits.class_3 = class_3 + hdr.output.logits.class_3;
	}

	action set_mm3_word_4(bit<32> class_0, bit<32> class_1, bit<32> class_2, bit<32> class_3) {
		ig_md.logits_partial.class_0 = class_0 + ig_md.logits_partial.class_0;
		ig_md.logits_partial.class_1 = class_1 + ig_md.logits_partial.class_1;
		ig_md.logits_partial.class_2 = class_2 + ig_md.logits_partial.class_2;
		//ig_md.logits_partial.class_3 = class_3 + ig_md.logits_partial.class_3;
	}

	action set_mm3_word_5(bit<32> class_0, bit<32> class_1, bit<32> class_2, bit<32> class_3) {
		hdr.output.logits.class_0 = class_0 + hdr.output.logits.class_0;
		hdr.output.logits.class_1 = class_1 + hdr.output.logits.class_1;
		hdr.output.logits.class_2 = class_2 + hdr.output.logits.class_2;
		//hdr.output.logits.class_3 = class_3 + hdr.output.logits.class_3;
	}

	action set_mm3_word_6(bit<32> class_0, bit<32> class_1, bit<32> class_2, bit<32> class_3) {
		ig_md.logits_partial.class_0 = class_0 + ig_md.logits_partial.class_0;
		ig_md.logits_partial.class_1 = class_1 + ig_md.logits_partial.class_1;
		ig_md.logits_partial.class_2 = class_2 + ig_md.logits_partial.class_2;
		//ig_md.logits_partial.class_3 = class_3 + ig_md.logits_partial.class_3;
	}

	action set_mm3_word_7(bit<32> class_0, bit<32> class_1, bit<32> class_2, bit<32> class_3) {
		hdr.output.logits.class_0 = class_0 + hdr.output.logits.class_0;
		hdr.output.logits.class_1 = class_1 + hdr.output.logits.class_1;
		hdr.output.logits.class_2 = class_2 + hdr.output.logits.class_2;
		//hdr.output.logits.class_3 = class_3 + hdr.output.logits.class_3;
	}

	table mm3_word_0 {
		key = {
			ig_md.hidden2.w0 : ternary;
		}
		actions = {
			set_mm3_word_0;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm3_word_1 {
		key = {
			ig_md.hidden2.w1 : ternary;
		}
		actions = {
			set_mm3_word_1;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm3_word_2 {
		key = {
			ig_md.hidden2.w2 : ternary;
		}
		actions = {
			set_mm3_word_2;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm3_word_3 {
		key = {
			ig_md.hidden2.w3 : ternary;
		}
		actions = {
			set_mm3_word_3;
			noaction;
		}
		size			= 8192;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm3_word_4 {
		key = {
			ig_md.hidden2.w4 : ternary;
		}
		actions = {
			set_mm3_word_4;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm3_word_5 {
		key = {
			ig_md.hidden2.w5 : ternary;
		}
		actions = {
			set_mm3_word_5;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm3_word_6 {
		key = {
			ig_md.hidden2.w6 : ternary;
		}
		actions = {
			set_mm3_word_6;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	table mm3_word_7 {
		key = {
			ig_md.hidden2.w7 : ternary;
		}
		actions = {
			set_mm3_word_7;
			noaction;
		}
		size			= 4096;
		default_action	= noaction();
		idle_timeout	= false;
	}

	apply {
		compute_flow_hash();

		ig_md.flow.other_flow   = swap_flow_sig.execute(ig_md.flow.hash[15:0]);
		ig_md.flow.gap          = swap_last_ts.execute(ig_md.flow.hash[15:0]);

		// New flow: another flow held the slot, or the gap reached the idle timeout.
		if (ig_md.flow.other_flow == 1 || ig_md.flow.gap[31:20] != 0) {
			restart_pkt_count.execute(ig_md.flow.hash[15:0]);
			restart_byte_count.execute(ig_md.flow.hash[15:0]);

			ig_md.feat.ts_rel       = 0;
			ig_md.feat.packet_index = 0;
			ig_md.feat.flow_len     = (bit<32>)hdr.ipv4.total_len;
		} else {
			ig_md.feat.packet_index = next_pkt_count.execute(ig_md.flow.hash[15:0]);
			ig_md.feat.flow_len     = next_byte_count.execute(ig_md.flow.hash[15:0]);
			ig_md.feat.ts_rel       = ig_md.flow.gap;
		}

		// MM1
		mm1_proto_ts_rel.apply();
		mm1_flags_pkt_idx.apply();
		mm1_flow_len.apply();

		ig_md.hidden1.w0 = ig_md.hidden1.w0 + ig_md.hidden1_partial.w0;
		ig_md.hidden1.w1 = ig_md.hidden1.w1 + ig_md.hidden1_partial.w1;
		ig_md.hidden1.w2 = ig_md.hidden1.w2 + ig_md.hidden1_partial.w2;
		ig_md.hidden1.w3 = ig_md.hidden1.w3 + ig_md.hidden1_partial.w3;
		ig_md.hidden1.w4 = ig_md.hidden1.w4 + ig_md.hidden1_partial.w4;
		ig_md.hidden1.w5 = ig_md.hidden1.w5 + ig_md.hidden1_partial.w5;
		ig_md.hidden1.w6 = ig_md.hidden1.w6 + ig_md.hidden1_partial.w6;
		ig_md.hidden1.w7 = ig_md.hidden1.w7 + ig_md.hidden1_partial.w7;
		ig_md.hidden1.w8 = ig_md.hidden1.w8 + ig_md.hidden1_partial.w8;
		ig_md.hidden1.w9 = ig_md.hidden1.w9 + ig_md.hidden1_partial.w9;
		ig_md.hidden1.w10 = ig_md.hidden1.w10 + ig_md.hidden1_partial.w10;
		ig_md.hidden1.w11 = ig_md.hidden1.w11 + ig_md.hidden1_partial.w11;
		ig_md.hidden1.w12 = ig_md.hidden1.w12 + ig_md.hidden1_partial.w12;
		ig_md.hidden1.w13 = ig_md.hidden1.w13 + ig_md.hidden1_partial.w13;
		ig_md.hidden1.w14 = ig_md.hidden1.w14 + ig_md.hidden1_partial.w14;
		ig_md.hidden1.w15 = ig_md.hidden1.w15 + ig_md.hidden1_partial.w15;

		// MM2
		mm2_direct.apply();
		mm2_leaf_1.apply();
		mm2_leaf_2.apply();
		mm2_leaf_3.apply();
		mm2_leaf_4.apply();
		mm2_leaf_5.apply();
		mm2_leaf_6.apply();
		mm2_leaf_7.apply();
		mm2_leaf_8.apply();
		mm2_leaf_9.apply();
		mm2_leaf_10.apply();
		mm2_leaf_11.apply();
		mm2_leaf_12.apply();
		mm2_leaf_13.apply();
		mm2_leaf_14.apply();
		mm2_leaf_15.apply();

		mm2_group_1_3.apply();
		mm2_group_4_6.apply();
		mm2_group_7_9.apply();
		mm2_group_10_12.apply();
		mm2_group_13_15.apply();

		ig_md.hidden2.w0 = ig_md.hidden2.w0 + ig_md.hidden2_partial.w0;
		ig_md.hidden2.w1 = ig_md.hidden2.w1 + ig_md.hidden2_partial.w1;
		ig_md.hidden2.w2 = ig_md.hidden2.w2 + ig_md.hidden2_partial.w2;
		ig_md.hidden2.w3 = ig_md.hidden2.w3 + ig_md.hidden2_partial.w3;
		ig_md.hidden2.w4 = ig_md.hidden2.w4 + ig_md.hidden2_partial.w4;
		ig_md.hidden2.w5 = ig_md.hidden2.w5 + ig_md.hidden2_partial.w5;
		ig_md.hidden2.w6 = ig_md.hidden2.w6 + ig_md.hidden2_partial.w6;
		ig_md.hidden2.w7 = ig_md.hidden2.w7 + ig_md.hidden2_partial.w7;

		// MM3
		mm3_word_0.apply();
		mm3_word_1.apply();
		mm3_word_2.apply();
		mm3_word_3.apply();
		mm3_word_4.apply();
		mm3_word_5.apply();
		mm3_word_6.apply();
		mm3_word_7.apply();

		hdr.output.logits.class_0 = hdr.output.logits.class_0 + ig_md.logits_partial.class_0;
		hdr.output.logits.class_1 = hdr.output.logits.class_1 + ig_md.logits_partial.class_1;
		hdr.output.logits.class_2 = hdr.output.logits.class_2 + ig_md.logits_partial.class_2;
		//hdr.output.logits.class_3 = hdr.output.logits.class_3 + ig_md.logits_partial.class_3;

		add_conf_threshold();
		compute_logit_margins();

		confidence_check.apply();

		hdr.output.setValid();

		ig_tm_md.ucast_egress_port = OUTPUT_PORT;
		ig_tm_md.bypass_egress = 1;

		hdr.ethernet.dst_addr = 0;
	}
}

Pipeline(SwitchIngressParser(),
		 SwitchIngress(),
		 SwitchIngressDeparser(),
		 EmptyEgressParser(),
		 EmptyEgress(),
		 EmptyEgressDeparser()) pipe;

Switch(pipe) main;
