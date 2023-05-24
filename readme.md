# Mixed Integer Linear Programming (MILP) for Logistic Planning

This repository contains code and analysis for a case study on using mixed integer linear programming (MILP) for logistic planning. The problem was to optimize the delivery schedule and routing for a company providing a quality testing service.

## Problem Description

The company needs to deliver tools from a depot to a set of customers using a fleet of vehicles. Each customer has a demand and each vehicle has a capacity. The objective is to minimize the total costs which include tool usage, vehicle costs and distance costs.

## Methodology

The problem was approached as a Parallel Machines Scheduling Problem for the daily request schedule followed by a Capacitated Vehicle Routing Problem (CVRP) and solved using a ILP and MILP model. For certain instances of the problem, the solution was enhanced by using a heuristic tool.


