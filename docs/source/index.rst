.. discretesampling documentation master file, created by
   sphinx-quickstart on Mon Sep 25 12:39:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to discretesampling's documentation!
============================================

discretesampling is a python framework to perform Bayesian sampling from
distributions over discrete variables.

The aim is to separate the algorithms used to sample from these distributions
from the problem-specific descriptions of the variables and distributions.

For example, a problem involving :py:class:`decision trees<discretesampling.domain.decision_tree>` requires the user
to provide:

* a definition of a :py:class:`decision tree<discretesampling.domain.decision_tree.tree>`
* a :py:class:`proposal distribution<discretesampling.domain.decision_tree.tree_distribution>` for moving between trees
* a :py:class:`initial proposal distribution<discretesampling.domain.decision_tree.tree_initial_proposal>` for sampling an
  initial set of trees
* a :py:class:`target<discretesampling.domain.decision_tree.tree_target>` (or posterior) distribution to sample from

A generic :py:class:`MCMC<discretesampling.base.algorithms.MCMC>` or
:py:class:`SMC<discretesampling.base.algorithms.SMC>` sampler can then be used to sample from the target distribution.

Contents
========
.. toctree::
   :maxdepth: 2

   discretesampling


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
