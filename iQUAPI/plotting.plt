set terminal epslatex color standalone size 12cm,6cm
set output "PlotOutput.tex"

set xrange [0:100]
set yrange [-0.1:1.5]
set ylabel 'probability of exciton existence'
set xlabel 'time'
set title 'Effect of spectral density on decoherence (E = $\frac{\hbar}{2} * 0.5$)'


plot "read_J_PT_iQUAPI_ohmic.out" using 1:2 t 'ohmic spectral density' with lines ,\
"read_J_PT_iQUAPI_none.out" using 1:2 t 'no spectral density' with lines
