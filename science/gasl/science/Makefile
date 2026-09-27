# Makefile für gasl.tex
LATEX   = pdflatex
BIBER   = biber
MAIN    = gasl
PYTHON  = python3

.PHONY: all plots pdf clean distclean

all: plots pdf

plots:
	cd $(dir $(abspath $(firstword $(MAKEFILE_LIST)))) && \
	$(PYTHON) generate_plots.py

pdf: plots
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBER) $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(LATEX) -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).bcf $(MAIN).blg \
	       $(MAIN).log $(MAIN).out $(MAIN).run.xml \
	       $(MAIN).toc $(MAIN).lof $(MAIN).lot

distclean: clean
	rm -f $(MAIN).pdf fig0*.pdf
