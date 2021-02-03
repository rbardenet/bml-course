# Illustration of sampling when K >> N.
# Julyan Arbel edits of original code by
# Copyright (C) 2015, Tamara Broderick
# www.tamarabroderick.com

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# useful for sampling from the Dirichlet distribution
# you might need to install it first
# install.packages("MCMCpack")
library(MCMCpack)

# maximum number of data points to draw
maxN = 1000

# use these parameters by default
K_default = 1000
a_default = 10/K_default
# note: default run with these parameters
# appears at the end

gen_largeK_diri <- function(K,a) {
# Illustrates cluster assignments using
# Dirichlet-distributed component probabilities
#
# Args:
#  K: Dirichlet parameter vector length
#  a: Dirichlet parameter (will be repeated K times)
#
# Returns:
#  Nothing
#
# Illustrates Dirichlet-distributed samples
# as a partition of the unit interval
# (cf. Kingman paintbox) and illustrates
# samples from this (random) distribution

	# make the Dirichlet draw
	rhomx = rdirichlet(1,rep(a,K))
	# various other useful forms of rho
	rho = as.vector(rhomx)
	rhomxt = t(rhomx)
	crho = c(0,cumsum(rho))

	# initialize bar colors so that
	# no components have been chosen yet
	# "grey" = not chosen, "blue" = chosen
	bar_colors = rep("grey",K)

	# special plot size for very horizontal fig
	# x11(width=8,height=3)

	for(N in 0:maxN) {
		# want the option to illustrate
		# before draws are made
		if(N > 0) {
		  ##################################################################
			# uniform draw to decide which component is chosen
			u = ??? # replace ??? with a uniform random draw
			##################################################################
			draw = max(which(crho < u))
	
			# update bar color of chosen component
			bar_colors[draw] = "blue"
		}

		# bar plot makes it easy to plot
		# probabilities one after another
	        barplot(rhomxt,
	                beside=FALSE,
	                horiz=TRUE,
	                col=bar_colors,
			width=0.7,
			ylim=c(0,1),
			main=bquote(rho~"~Dirichlet"  # ~"("~.(a)~",...,"~.(a)~")"
                                ~", K="~.(K)~", N="~.(N))
       	        )

		if(N > 0) {
			# illustrate the uniform random variable
			points(u, 0.9, pch=25, col="red", bg="red")
		}
	
		# Generate a new draw for each press of "enter"
                # Press 'x' when finished
	        line <- readline()
	        if(line == "x") {
			dev.off()
			return("done")
		}
	}
}	

# default run with default parameters
gen_largeK_diri(K_default, a_default)


