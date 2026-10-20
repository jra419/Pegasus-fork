#ifndef _DEPARSER_
#define _DEPARSER_

#include "headers.p4"

// ---------------------------------------------------------------------------
// Ingress Deparser
// ---------------------------------------------------------------------------

control SwitchIngressDeparser(packet_out pkt,
							  inout header_t hdr,
							  in metadata_t ig_md,
							  in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.tcp);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.output);
    }
}

#endif
