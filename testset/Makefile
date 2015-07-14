DATADIR = data
RESULTDIR = results
DIAGDIR = diagnostics
PLOTDIR = plots
LOGLEVEL = info
LOGFILE = N
ifeq ($(LOGFILE), Y)
	LOGFILEARG = --logfile=$(RESULTDIR)/$*.log
else
	LOGFILEARG = 
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

names = PTF09fox_B PTF09fox_R \
	SN2006ob_B SN2006ob_R \
	SNF20060512-002_B SNF20060512-002_R \
	SNF20060609-002_B SNF20060609-002_R \
	SNF20061009-008_B SNF20061009-008_R \
	SNF20070429-000_B SNF20070429-000_R \
	SNF20070831-015_B SNF20070831-015_R \
	SNF20070902-021_B SNF20070902-021_R \
	SNF20080707-012_B SNF20080707-012_R \
	SNF20080717-000_B SNF20080717-000_R \
	SNF20080720-001_B SNF20080720-001_R \
	SNF20080725-004_B SNF20080725-004_R

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

$(foreach name,$(names),$(name)-plots) : %-plots : $(PLOTDIR)/%_timeseries.eps $(PLOTDIR)/%_waveslice.eps

$(PLOTDIR)/%_timeseries.eps $(PLOTDIR)/%_waveslice.eps :
	mkdir -p $(PLOTDIR)
	cubefit-plot \
	$(DATADIR)/$*/$*.json \
	$(RESULTDIR)/$*.fits \
	$(PLOTDIR)/$* \
	--dataprefix=$(DATADIR)/$* \
	$(IDRFILESARG) \
	--diagdir=$(DIAGDIR)/$*

clean :
	rm $(RESULTSDIR)/*

clean-plots :
	rm $(PLOTDIR)/*
