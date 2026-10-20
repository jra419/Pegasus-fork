PROG		   := mlp-b-st.p4
ARCH		   := P4_ARCHITECTURE=t2na
VERSION		   := P4_VERSION=p4_16
P4C			   := $(SDE_INSTALL)/bin/bf-p4c
DEBUG_FLAGS	   := -g --verbose 2 --create-graphs -b tofino2
MKDIR		   := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR	   := $(MKDIR)/build
INSTALL_SCRIPT	= $(MKDIR)/../../build_tools/p4_build.sh

debug: $(PROG)
	$(P4C) $(DEBUG_FLAGS) -o $(BUILD_DIR) $(PROG)

install: $(PROG)
	$(INSTALL_SCRIPT) $(PROG) $(ARCH) $(VERSION)

.PHONY: clean

clean:
	rm -rf build
