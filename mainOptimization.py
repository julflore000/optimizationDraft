from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import array
import shutil
import sys

#define respective weights in another file or call in
# alpha = .1
# beta = .5
# gamma = .2


#create model
model = ConcreteModel()

############ START: SETS ############
#hypothetical renewable plants
model.R = Set(initialize=getRePlantLocations(), doc='Renewable plant locations')

#coal plants looking to be retired (should be preprocesed)
model.C = Set(initialize = getCoalPlants(),doc='Coal plant IDs')

#years of analysis
timeFrame = 100
model.Y = Set(initialize= np.arange(0,timeFrame))
############ END: SETS ############


############ START: PARAMETERS ############
#historical generation of each coal plant on an hourly basis
model.histGen = Param(model.C,model.Y,initialize = getHistoricalGeneration(),doc = "Historical Generation of coal plants")

#capacity factor parameter for RE sites
model.capFactor = Param(model.R,model.Y, initialize = getRECapacityFactors(),doc = "Hourly CF of each RE location")

#CAPEX for REs
model.ReCapex = Param(model.R,model.Y, initialize = getReCapex(),doc = "RE plants CAPEX values ($/MW")

#OPEX for REs
model.ReOpex = Param(model.R,model.Y, initialize = getReOpex(),doc = "RE plants OPEX values $/MWh")

#OPEX for Coal- don't need hourly set as coal OPEX values don't have any decline factors
model.CoalOpex = Param(model.C,initialize = getCoalOpex(), doc = "Coal plants OPEX values $/MWh")

#max capacity for REs at that site
model.maxCap = Param(model.R, initialize = getMaxReCap(), doc ='Maximum capacity aviable at the site for wind & solar')

#how many sites we can place for each coal plant
model.maxReSites = Param(model.C, initialize = getMaxReSites(), doc = "How many sites can we put at each coal plant")

#health impact of each coal plant
model.healthImpacts = Param(model.C, initialize = getCoalHealthImpacts(), doc = "Health impacts of each coal plant loc- from Isaac")

#if you retire a coal plant it can create x jobs/MW
model.CoalRetirementEF = Param(model.C, initialize = getCoalRetirementEfs(), doc="How many jobs are created for retireing each coal plant-most likely just a single value")

#coal O&M jobs- only looking at plant level jobs- does not include miners
model.CoalOperationsEF = Param(model.C,initialize = getCoalEF(),doc="How many jobs/MW are there in O&M of a plant")

#jobs created by constructing a RE plant (job-years/MW), need to include hourly dependence as well
#will need to read in getREEFs which will give us construction & O&M jobs, then filter out in correc params
model.ReConEF = Param(model.R,model.Y,initialize = getReEFConstruction(),doc = "How many construction jobs are created for a RE plant (job-years)/MW")
model.ReOperationsEF = Param(model.R,model.Y,initialize = getReEFO&M(),doc = "How many operations and maintenance jobs are created for a RE plant (job-years)/MW")

############ END: PARAMETERS ############

############ START: VARIABLES ############
#the capacity to be invested for that renewable plant to replace coal
model.capInvest = Var(model.R,model.C,model.Y, within= NonNegativeReals, doc = "Capacity to be invested in that renewable plant to replace coal")

#total amount of coal capacity to retire in that year
model.capRetire = Var(model.C,model.Y, within= NonNegativeReals,doc = "amount of capacity to be retired for each coal plant")

#generation for each renewable plant for each coal plantin that year
model.reGen = Var(model.R,model.C,model.Y, within = NonNegativeReals, doc = "RE generation for each plant")

#generation for each coal plant in that year
model.coalGen = Var(model.C,model.Y, within = NonNegativeReals, doc = "Coal generation for each plant")

#overall capacity size of each RE plant for each coal plant tie
model.reCap = Var(model.R,model.C,model.Y, within = NonNegativeReals, doc = "Capacity size for each RE plant")

#build/invest in that RE plant to replace that coal plant for that year
model.reInvest = Var(model.R,model.C,model.Y, within = Binary, doc = "Variable to invest in RE to replace coal")

#Retire (1) or not retire (0) that coal plant for that year
model.coalRetire = Var(model.C,model.Y, within = Binary, doc = "Variable to retire coal plant")

#renewable plant online (contributing energy)
model.reOnline = Var(model.R,model.C,model.Y,within = Binary, doc = "Whether that renewable plant to replace coal is online or not")

#indicator for whether coal plant is on or off
model.coalOnline = Var(model.C,model.Y, within = Binary, doc = "Whether the coal plant is operating (1) or not operating (0)")
############ END: VARIABLES ############

############ START: CONSTRAINTS ############
#generation of each coal plant must equal historical generation * whether that plant is online or not
def coalGenRule(model,c,y):
    return(model.coalGen[c,y] == model.histGen[c,y]*model.coalOnline[c,y])
model.coalGenConstraint = Constraint(model.C,model.Y, rule=coalGenRule, doc='Coal generation must equal historical generation * whether that plant is online')

#generation of each RE plant must be less than or equal to capacity factor* chosen capacity
def reGenRule(model,r,c,y):
    return(model.reGen[r,c,y] <= model.capFactor[r,c,y]*model.reCap[r,c,y])
model.reGenConstraint = Constraint(model.R,model.C,model.Y, rule=reGenRule, doc='RE generation must be less than or equal to capacity factor* chosen capacity')

#renewable generation must match displaced coal generation in that location for that year
def reGenBalanceRule(model,c,y):
    return(sum(model.reGen[r,c,y] for r in model.R) == model.histGen[c,y] - model.coalGen[c,y])
model.reGenBalanceConstraint = Constraint(model.C,Model.R,rule=reGenBalanceRule, doc = "RE generation for each coal location must equal retired capacity")

#capacity of RE plant must be less than or equal to max capacity at that site * whether the plant is online for investment
def reCapRule(model,r,c,y):
    return(model.reCap[r,c,y] <= model.maxCap[r]*model.reOnline[r,c,y])
model.reCapConstraint = Constraint(model.R,model.C,model.Y, rule = reCapRule, doc = "renewable capacity decision variable should be less then or equal to max capacity* whether investment is allowed")

#sum of total RE energy provided to coal plants must be less then total available max capacity of that single RE plant (don't want to overcount)
def reCapLimitRule(model,r,c,y):
    return(sum(model.reGen[r,c,y] for c in model.C) <= model.maxCap[r])
model.reCapLimitConstraint = Constraint(model.R,model.C,model.Y, rule = reCapLimitRule, doc = "Constraint prevents a single RE location from building more then max capacity to neighboring coal locations")

#capacity able to be invested in each year is difference between current RE cap and prior year RE cap
def capInvestRule(model,r,c,y):
    if y == model.Y[0]: # I believe this should be zero based indexing however this could be wrong
        return(model.reInvest[r,c,y] == model.reCap[r,c,y])
    return(model.reInvest[r,c,y] == model.reCap[r,c,y]-model.reCap[r,c,y-1])
model.reCapInvestConstraint = Constraint(model.R,model.C,model.Y, rule = capInvestRule, doc = "RE capacity to invest must be equal to difference in RE cap across years")

#capacity able to be invested must be less than or equal to max capacity for that plant*whether we decided to invest
def capInvestLimitRule(model,r,c,y):
    return(model.reInvest[r,c,y] == model.maxCap[r]*model.reInvest)
model.capInvestMaxConstraint = Constraint(model.R,model.C,model.Y, rule = capInvestLimitRule, doc = "RE capacity to invest must be less than or equal to max cap * whether we invest or not")

#decision to retire RE is current online - prior year online status
def reInvestRule(model,r,c,y):
    if y == model.Y[0]:
        return model.reInvest[r,c,y] == model.reOnline[r,c,y]
    #else
    return model.reInvest[r,c,y] == model.reOnline[r,c,y] - model.reOnline[r,c,y-1]
model.reInvestConstraint = Constraint(model.R,model.C,model.Y,rule=reInvestRule,doc= "Decision to invest in RE is current year - prior")

#number of RE sites to invest in is less than or equal to maxSites for each coal plant*whether we retire coal
def reInvestSiteLimitRule(model,c,y):
    return(sum(model.reInvest[r,c,y] for r in model.R) <= model.maxReSites[c]*model.coalRetire[c,y])
model.reSiteLimitConstraint = Constraint(model.C,model.Y, rule = reInvestSiteLimitRule,doc = "Number of new RE sites must be less than or equal to max RE sites for that coal plant * whether we retire")
#coal retire is current year - prior year
def coalRetireRule(c,y):
    if y == model.Y[0]: # I believe this should be zero based indexing however this could be wrong
        return(model.coalRetire[c,y] == model.coalOnline[c,y])
    #else
    return(model.coalRetire[c,y] <= model.coalOnline[c,y-1] - model.coalOnline[c,y])
model.coalRetireConstraint = Constraint(model.C,model.Y, rule = coalRetireRule, doc = "Coal retire activation is current year must prior year")

#limit retirement to only once
def coalRetireLimitRule(c):
    return(sum(model.coalRetire[c,y] for y in model.Y) <= 1)
model.coalRetireLimitConstraint = Constraint(model.C, rule = coalRetireRule, doc = "Can only retire a coal plant once over time period")

############ END: CONSTRAINTS ############

############ START: OBJECTIVE ############
#return the overall system costs of each plant
def systemCosts(model):
    #can also rewrite final below statement as for re portion
    '''    totalSystemCost = 0
    for year in model.Y:
        for rePlant in model.R:
            for coalPlant in model.C:
                totalSystemCost += model.ReCapex[rePlant,year]*model.capInvest[rePlant,coalPlant,year] + model.ReOpex[rePlant,year]*model.reGen[rePlant,coalPlant,year]
    return(totalSystemCost)'''
    return (sum(sum(sum(model.ReCapex[r,y]*model.capInvest[r,c,y] + model.ReOpex[r,y]*model.reGen[r,c,y] for c in model.C)for r in model.R) for y in model.Y)\
        + sum(sum(model.CoalOpex[c,y]*model.coalGen[c,y] for c in model.C)for y in model.Y))

#return helathimpacts of coal plants
def healthCosts(model):
    return(sum(sum(model.healthImpacts[c]*model.coalGen[c,y] for  c in model.C)for y in model.Y))

#returns total jobs in system: coalretirement EF * coalRetire capacity + current coal jobs + RE construction and operations jobs
def jobImpact(model):
    return (sum(sum(model.CoalRetirementEF[c]*model.capRetire[c,y] + model.CoalOperationsEF[c]*model.coalGen[c,y] for c in model.C)for y in model.Y)\
        + sum(sum(sum(model.ReConEF[r,y]**model.capInvest[r,c,y] + model.ReOperationsEF[r,y]*model.reGen[r,c,y] for c in model.C)for r in model.R)for y in model.Y))

#return the system costs + helathCosts - jobImpact (want to minimize first two but then maxmize job impacts)with respective multipliers on each
def objectiveRule(model):
    return(alpha*systemCosts(model) + beta*healthCosts(model)- gamma* jobImpact(model))
model.objective = Objective(rule=objectiveRule, sense=minimize, doc='Minimize system costs, health damages, while maximizing jobs')

############ END: OBJECTIVE ############
