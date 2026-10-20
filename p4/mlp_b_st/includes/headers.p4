#ifndef _HEADERS_
#define _HEADERS_

typedef bit<48> mac_addr_t;
typedef bit<32> ipv4_addr_t;
typedef bit<128> ipv6_addr_t;
typedef bit<12> vlan_id_t;


typedef bit<16> ether_type_t;
const ether_type_t ETHERTYPE_IPV4 = 16w0x0800;
const ether_type_t ETHERTYPE_ARP = 16w0x0806;
const ether_type_t ETHERTYPE_IPV6 = 16w0x86dd;
const ether_type_t ETHERTYPE_VLAN = 16w0x8100;

typedef bit<8> ip_protocol_t;
const ip_protocol_t IP_PROTOCOLS_ICMP = 1;
const ip_protocol_t IP_PROTOCOLS_TCP = 6;
const ip_protocol_t IP_PROTOCOLS_UDP = 17;

header ethernet_h {
    mac_addr_t dst_addr;
    mac_addr_t src_addr;
    bit<16> ether_type;
}

header ipv4_h {
    bit<4> version;
    bit<4> ihl;
    bit<8> diffserv;
    bit<16> total_len;
    bit<16> identification;
    bit<3> flags;
    bit<13> frag_offset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdr_checksum;
    ipv4_addr_t src_addr;
    ipv4_addr_t dst_addr;
}

header tcp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<32> seq_no;
    bit<32> ack_no;
    bit<4>  data_offset;
    bit<4>  res;
    bit<8>  flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgent_ptr;
}

header udp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> hdr_length;
    bit<16> checksum;
}

// Hidden-layer values as 32-bit words, each holding two 13-bit lanes (word j = values 2j, 2j+1).
struct words16_t {
    bit<32> w0;
    bit<32> w1;
    bit<32> w2;
    bit<32> w3;
    bit<32> w4;
    bit<32> w5;
    bit<32> w6;
    bit<32> w7;
    bit<32> w8;
    bit<32> w9;
    bit<32> w10;
    bit<32> w11;
    bit<32> w12;
    bit<32> w13;
    bit<32> w14;
    bit<32> w15;
}

struct words8_t {
    bit<32> w0;
    bit<32> w1;
    bit<32> w2;
    bit<32> w3;
    bit<32> w4;
    bit<32> w5;
    bit<32> w6;
    bit<32> w7;
}

// Leaf index (0..15) chosen in each MM2 codeblock 1..15.
struct mm2_leaves_t {
    bit<4> cb1;
    bit<4> cb2;
    bit<4> cb3;
    bit<4> cb4;
    bit<4> cb5;
    bit<4> cb6;
    bit<4> cb7;
    bit<4> cb8;
    bit<4> cb9;
    bit<4> cb10;
    bit<4> cb11;
    bit<4> cb12;
    bit<4> cb13;
    bit<4> cb14;
    bit<4> cb15;
}

// Per-class logits, one unpacked 32-bit value each; only the model's classes are written.
struct logits_t {
    bit<32> class_0;
    bit<32> class_1;
    bit<32> class_2;
    bit<32> class_3;
    bit<32> class_4;
    bit<32> class_5;
    bit<32> class_6;
    bit<32> class_7;
}

header output_h {
    logits_t logits;
    bit<8>   confident;     // 1 if the top logit beats the second by at least the threshold
}

struct header_t {
    ethernet_h ethernet;//112
    ipv4_h ipv4;//160
    tcp_h tcp;//160
    udp_h udp;//64
    output_h output;//264
}

// MM1 input features computed on the switch, exactly as models/mlp_b/mlp_b_st/pcap_features.py
// does (widths of TOFINO_CODEBLOCK_BIT_LENGTHS); ip_proto is read from hdr.ipv4.protocol.
struct features_t {
    bit<8>  tcp_flags;      // 0 if not TCP
    bit<32> ts_rel;         // time since the flow's previous packet, 2^8 ns units; 0 if new
    bit<16> packet_index;   // position in the flow, saturating
    bit<32> flow_len;       // sum of the flow's IP total lengths, this packet included, saturating
}

// Per-packet scratch for the flow-state registers.
struct flow_meta_t {
    bit<16> port_src;       // 0 if not TCP/UDP
    bit<16> port_dst;
    bit<32> hash;           // CRC-32 of the 5-tuple, as zlib.crc32: [15:0] register index,
                            // [31:16] flow signature
    bit<32> gap;            // time since the slot's previous packet, 2^8 ns units (from last_ts)
    bit<1>  other_flow;     // the slot held another flow's signature
}

struct empty_header_t {}

struct empty_metadata_t {}

// Confidence check scratch: t_j = logit_j + threshold, d_ij = logit_i - t_j.
struct confidence_t {
    bit<32> t0;
    bit<32> t1;
    bit<32> t2;
    bit<32> d01;
    bit<32> d02;
    bit<32> d10;
    bit<32> d12;
    bit<32> d20;
    bit<32> d21;
}

struct metadata_t {
    flow_meta_t     flow;               // flow-state scratch
    features_t      feat;               // MM1 input features
    words16_t       hidden1;            // MM1 output
    words16_t       hidden1_partial;    // MM1 partial sum, added into hidden1
    mm2_leaves_t    mm2_leaf;           // MM2 leaf index per codeblock
    words8_t        hidden2;            // MM2 output
    words8_t        hidden2_partial;    // MM2 partial sum, added into hidden2
    logits_t        logits_partial;     // MM3 partial sum, added into hdr.output.logits
    confidence_t    conf;               // confidence check scratch
}

#endif /* _HEADERS_ */
