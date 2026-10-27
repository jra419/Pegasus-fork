#ifndef _PARSER_
#define _PARSER_

#include "headers.p4"

// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------

parser SwitchIngressParser(packet_in pkt,
						   out header_t hdr,
						   out metadata_t ig_md,
						   out ingress_intrinsic_metadata_t ig_intr_md) {

	TofinoIngressParser() tofino_parser;

	state start {
		tofino_parser.apply(pkt, ig_intr_md);
		transition parse_ethernet;
	}

	state parse_ethernet {
		pkt.extract(hdr.ethernet);
		transition select(hdr.ethernet.ether_type) {
			ETHERTYPE_IPV4: parse_ipv4;
			default: reject;
		}
	}

	state parse_ipv4 {
		pkt.extract(hdr.ipv4);
		ig_md.flow.port_src		= 0;
		ig_md.flow.port_dst		= 0;
		ig_md.feat.tcp_flags	= 0;
		transition select(hdr.ipv4.ihl, hdr.ipv4.frag_offset, hdr.ipv4.protocol) {
			(5, 0, IP_PROTO_TCP) : parse_tcp;
			(5, 0, IP_PROTO_UDP) : parse_udp;
			default: accept;
		}
	}

	state parse_tcp {
		pkt.extract(hdr.tcp);
		ig_md.flow.port_src		= hdr.tcp.src_port;
		ig_md.flow.port_dst		= hdr.tcp.dst_port;
		ig_md.feat.tcp_flags	= hdr.tcp.flags;
		transition accept;
	}

	state parse_udp {
		pkt.extract(hdr.udp);
		ig_md.flow.port_src = hdr.udp.src_port;
		ig_md.flow.port_dst = hdr.udp.dst_port;
		transition accept;
	}
}

#endif
