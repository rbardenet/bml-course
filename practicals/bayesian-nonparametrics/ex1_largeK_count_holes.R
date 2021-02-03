# Illustration of cluster growth
# given a Dirichlet-distributed vector
# with K >> N.
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

# useful for sampling from a Dirichlet distribution
# you might need to install it first
# install.packages("MCMCpack")
library(MCMCpack)

# maximum number of data points to draw
maxN <- 1000

# use these parameters by default
K_default <- 1000
a_default <- 10 / K_default
# note: default run with these parameters
# appears at the end

gen_largeK_count <- function(K, a) {
  # Illustrates how number of clusters changes
  # with Dirichlet-distributed component probabilities
  #
  # Args:
  #  K: Dirichlet parameter vector length
  #  a: Dirichlet parmeter (will be repeated K times)
  #
  # Returns:
  #  Nothing

  # make the Dirichlet draw
  rhomx <- rdirichlet(1, rep(a, K))
  # another useful form of rho
  rho <- as.vector(rhomx)

  # records which clusters have been sampled so far
  uniq_draws <- c()
  # cluster samples in order of appearance (ooa)
  ooa_clust <- c()

  for (N in 1:maxN) {
    ############################################################
    # draw a cluster assignment from the components
    draw <- ??? # replace ??? by the sampling of a new cluster assignment
    ############################################################
    # update info about cluster draws
    uniq_draws <- unique(c(uniq_draws, draw))
    ooa <- which(draw == uniq_draws)
    ooa_clust <- c(ooa_clust, ooa)

    # plot cluster assignments in order of appearance
    plot(
      x = seq(1, N),
      ############################################################
      y = ???, # replace ??? by what we actually want to see on the y-axis
      ############################################################
      xlab = "Sample index",
      ylab = "Cluster by order of appearance",
      ylim = c(0, max(10, length(uniq_draws))),
      xlim = c(0, max(10, N)),
      pch = 19,
      main = bquote(rho ~ "~Dirichlet" # ~"("~.(a)~",...,"~.(a)~")"
      ~", K=" ~ .(K))
    )

    # Generate a new draw for each press of "enter"
    # Press 'x' when finished
    line <- readline()
    if (line == "x") {
      return("done")
    }
  }
}

# default run with default parameters
gen_largeK_count(K_default, a_default)

# You can try different parameter combinations
gen_largeK_count(K=1000,a=0.01)
