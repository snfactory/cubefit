DATADIR = data
RESULTDIR = results
DIAGDIR = diagnostics
PLOTDIR = plots
LOGLEVEL = info
LOGFILE = N
PLOTEPOCHS = N
ifeq ($(LOGFILE), Y)
	LOGFILEARG = --logfile=$(RESULTDIR)/$*.log
else
	LOGFILEARG = 
endif
ifeq ($(PLOTEPOCHS), Y)
	PLOTEPOCHSARG = --plotepochs
else
	PLOTEPOCHSARG = 
endif

# user override for --idrfiles argument in `cubefit-plot`
ifeq (exists, $(shell [ -e ./Make.user ] && echo exists ))
include ./Make.user
endif
ifdef IDRPREFIX
	IDRFILESARG = --idrfiles $(IDRPREFIX)/*/$(firstword $(subst _, ,$*))/$(firstword $(subst _, ,$*))_*_$(word 2,$(subst _, ,$*)).fits
else
	IDRFILESARG =
endif

names = LSQ12dbr_R \
	PTF09fox_B PTF09fox_R \
	PTF10ndc_B \
	PTF10nlg_B \
	SN2005ki_B SN2005ki_R \
	SN2006ac_B \
	SN2006ob_B SN2006ob_R \
	SN2011by_R \
	SNF20050919-000_B \
	SNF20051003-003_R \
	SNF20060512-002_B SNF20060512-002_R \
	SNF20060609-002_B SNF20060609-002_R \
	SNF20060624-019_R \
	SNF20061009-008_B SNF20061009-008_R \
	SNF20070427-001_R \
	SNF20070429-000_B SNF20070429-000_R \
	SNF20070504-017_R \
	SNF20070712-003_R \
	SNF20070831-015_B SNF20070831-015_R \
	SNF20070902-021_B SNF20070902-021_R \
	SNF20080707-012_B SNF20080707-012_R \
	SNF20080717-000_B SNF20080717-000_R \
	SNF20080720-001_B SNF20080720-001_R \
	SNF20080725-004_B SNF20080725-004_R \
	SNF20080821-000_R \
	SNF20080918-000_R \
	SNNGC4424_R \
	SNNGC6801_B SNNGC6801_R

$(names) : % : $(RESULTDIR)/%.fits

all: $(names)

all-plots : $(foreach name,$(names),$(name)-plots)

$(RESULTDIR)/%.fits :
	mkdir -p $(RESULTDIR)
	mkdir -p $(DIAGDIR)/$*
	cubefit $(DATADIR)/$*/$*.json \
        $(RESULTDIR)/$*.fits \
        --dataprefix=$(DATADIR)/$* \
        --diagdir=$(DIAGDIR)/$* \
        --loglevel=$(LOGLEVEL) $(LOGFILEARG) \
        --mu_wave=0.07 \
        --mu_xy=0.001

$(foreach name,$(names),$(name)-plots) : %-plots :
	mkdir -p $(PLOTDIR)/$*
	cubefit-plot \
	$(DATADIR)/$*/$*.json \
	$(RESULTDIR)/$*.fits \
	$(PLOTDIR)/$*/$* \
	--dataprefix=$(DATADIR)/$* \
	$(IDRFILESARG) \
	--diagdir=$(DIAGDIR)/$* $(PLOTEPOCHSARG)

clean :
	rm $(RESULTSDIR)/*

clean-plots :
	rm $(PLOTDIR)/*
