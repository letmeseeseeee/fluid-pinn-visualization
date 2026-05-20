build:
	xelatex -file-line-error -interaction=nonstopmode -synctex=1 main.tex
	bibtex main
	xelatex -file-line-error -interaction=nonstopmode -synctex=1 main.tex
	xelatex -file-line-error -interaction=nonstopmode -synctex=1 main.tex

clean:
	rm -rf *.pdf *.log *.aux *.bbl *.blg *.out *.synctex.gz *.toc
