# Illustration of a Pitman Yor process 
# Gaussian mixture model
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

# putting a limit on the looping
maxiters = 1000

# note: function call made with default
# parameter settings at the end

dpmm_sample <- function(
	alpha = 2,
	mu0 = 0,
	sig_0 = 1.5,
	sig = 0.3
	) {
# Illustrates using the Dirichlet process
# mixture model as a generative model
#
# Args:
#  alpha: DP-stick-breaking/GEM parameter
#  mu0: normal mean for cluster param prior (repeated twice)
#  sig_0: normal sd for cluster param prior (covar is diag with this value)
#  sig: nromal sd for points in a cluster (covar is diag with this value)
#
# Returns:
#  Nothing

	# some colors for the points in a cluster
	palette(rainbow(10))
	# set up the figure into two sized figures
	par(mar = rep(2,4))
	layout(matrix(c(1,2), 2, 1, byrow=TRUE),
		heights=c(1,4)
		)

	# initializations
	rho = c()  # GEM-generated probabilities
	rhosum = 0  # sum of probabilities generated so far
	csum = c(0)  # cumulative sums of probs gen'd so far
	bar_colors = c()  # colors to distinguish clusters
	x = c()  # points generated from the DPMM
	z = c()  # cluster assignments generated in the DPMM
	mu = c()  # cluster means generated in the DPMM
	N = 0	# number of data points generated so far
	newN = 0  # number of data points to generate next round (user input)

	for(iter in 1:maxiters) {
		# generate new DPMM data points
		if(newN > 0) {
		for(n in 1:newN) {
			# uniform draw that decides which component is chosen
			u = runif(1)
			
			# instantiate components until the chosen
			# one is reached
			while(rhosum < u) {
			  ###############################################################
			  # beta stick-break for PY
				V = ???
			  ###############################################################
				# mass resulting from stick-break
				newrho = (1-rhosum)*V
				
				# update list of instantiated probabilities
				rho = rbind(rho,newrho)
				rhosum = rhosum + newrho
				csum = c(csum, rhosum)

				# fill in bars on the plot for
				# instantiated probabilities
				bar_colors = c(bar_colors,"grey")	

				# generate new cluster means for
				# instantiated probabilities		
				newmu = rnorm(2,mu0,sig_0)
				mu = rbind(mu, newmu)		
			}

			# decide which cluster was chosen for this data point
			thisz = max(which(csum < u))
			z = c(z,thisz)
			# generate a data point in this cluster
			thismu = mu[thisz,]
			newx = rnorm(cbind(1,1),thismu,cbind(sig,sig))
			x = rbind(x,newx)
		
		}
		}
		# record correct number of data points
		N = N + newN

		# plot the component probabilities instantiated so far
		barplot(rbind(rho,1-rhosum),
			beside=FALSE,
			horiz=TRUE,
			col=c(bar_colors, "white"),  # remaining mass in (0,1)
			ylim=c(0,1),
			width=0.7,
			main=bquote(rho~"~GEM("~.(alpha)~")")
			)

		# after initial display, plot how the next 
		# component is chosen
		if(N > 0) {
        	        points(u, 1, pch=25, col="red", bg="red")
        	}

		# just gets the labels and figure limits right
		# real plot still to come
		plot(x,
			pch=".",
			xlim=c(-5,5),
			ylim=c(-5,5),
			main=paste("N = ", toString(N),
				", #means = ", toString(length(rho)),
				", #clust = ", toString(length(unique(z))),sep="")
			)
	
		# plot all the instantiated means
		points(mu,
			pch=15,
			col="black"
			)

		# plot the data points generated from the DPMM thus far
		points(x,
			pch=19,
			col=z
			)

		# Generate one new draw for each press of "enter".
		# Writing a number generates that many new samples.
                # Press 'x' when finished
		line <- readline()
		if(line == "x") {
			return("done")
		} else if(line == "") {
			newN = 1
		} else {
			newN = as.numeric(line)
		}
	}

}

# default run with default parameters
dpmm_sample()


