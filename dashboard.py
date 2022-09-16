import streamlit as st
import pandas as pd
from pulp import *
import plotly.express as px

# Set the page
st.set_page_config(page_title="Test copy", page_icon="Random", layout="wide")

st.title("Simple Calculator")

# Import dataframe containing ar and mr data
mr = pd.read_csv("mr_data.csv")
ar = pd.read_csv("ar_data.csv")

# Initialize list containing compounds that will be calculated
compound = ["NiO", "CoO", "Fe2O3", "SiO2", "CaO", "MgO", "Al2O3", "P2O5", "Cr2O3", "SO2", "LOI", "H2O"]

# Function for chart
def pie_chart(data, value, name, color, title):
    fig = px.pie(data, values=value, names=name, color=color, width=600, height=600)
    fig.update_layout(title={ 'text': title,
                              'y':0.97,
                              'x':0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                      title_font_size=24)
    fig.add_annotation(dict(font=dict(size=20),
                            x=0.35,
                            y=-0.05,
                            showarrow=False,
                            text="Total: {} tonnes".format(round(sum(data[value]), 2)),
                            xanchor='left'))
    return fig

# Set interface
tab1, tab2, tab3, tab4 = st.tabs(["ore input", "rotary dryer", "rotary kiln", "electric arc furnace"])

with tab1:
	# Input initial value for calculation
	st.header("Ore Composition in Percent (%)")
	col1, col2 = st.columns(2)

	# Insert ore composition
	with col1:
		global wop
		wop = pd.DataFrame() # wop = wet ore percentage
		wop.loc[0, "NiO"] = st.number_input("NiO (%)", min_value=0.00)
		wop.loc[0, "CoO"] = st.number_input("CoO (%)", min_value=0.00)
		wop.loc[0, "Fe2O3"] = st.number_input("Fe2O3 (%)", min_value=0.00)
		wop.loc[0, "SiO2"] = st.number_input("SiO2 (%)", min_value=0.00)
		wop.loc[0, "CaO"] = st.number_input("CaO (%)", min_value=0.00)
		wop.loc[0, "MgO"] = st.number_input("MgO (%)", min_value=0.00)

	with col2:
		wop.loc[0, "Al2O3"] = st.number_input("Al2O3 (%)", min_value=0.00)
		wop.loc[0, "P2O5"] = st.number_input("P2O5 (%)", min_value=0.00)
		wop.loc[0, "Cr2O3"]  = st.number_input("Cr2O3 (%)", min_value=0.00)
		wop.loc[0, "SO2"] = st.number_input("SO2 (%)", min_value=0.00)
		wop.loc[0, "LOI"] = st.number_input("LOI (%)", min_value=0.00)
		wop.loc[0, "H2O"] = st.number_input("H2O (%)", min_value=0.00)



	# Warning indicator
	wop["Total"] = wop.sum(axis=1)
	st.write(wop.loc[0, "Total"])

	if wop.loc[0, "Total"] > 100:
		st.warning("Composition total exceeded 100%")
	elif wop.loc[0, "Total"] < 100:
		st.warning("Composition total less than 100%")
	
	st.header("Ore Weight (wet)")
	global wet_ore
	col1, col2 = st.columns([7,1])
	with col1:
		wet_ore = st.number_input("Insert ore weight in tonnes", min_value=0.00)
	with col2:
		st.write("\n")
		st.write("\n")
		dry_ore = st.checkbox("Calculate Dry Ore")

	# Calculating dry ore weight and composition
	col1, col2 = st.columns(2)
	with col1:
		if dry_ore is True:
			st.header("Total Wet Ore Weight (ton)")
			global wow
			wow = pd.DataFrame() # wow = wet ore weight
			for i in wop.columns:
				if i != "Total":
					wow[i] = wop[i] * wet_ore / 100
				else:
					break

			wow["Total"] = wow.sum(axis=1)
			st.dataframe(wow, width=2160, height=50)
			st.header("Wet Ore Composition in Percent (%)")
			st.dataframe(wop, width=2160, height=50)

	with col2:
		if dry_ore:
			st.header("Total Dry Ore Weight (ton)")
			global dow
			dow = pd.DataFrame() # dow = dry ore weight
			for i in wop.columns:
				if i != "Total" and i !="H2O":
					dow[i] = wop[i] * wet_ore / 100
				else:
					break
						
			dow["Total"] = dow.sum(axis=1)
			st.dataframe(dow, width=1080, height=50)

			st.header("Dry Ore Composition in Percent (%)")
			global dop
			dop = pd.DataFrame() # dop = dry ore percentage
			for i in dow.columns:
				if i != "Total":
					dop[i] = dow[i] / dow["Total"] * 100
				else:
					break
			dop["Total"] = dop.sum(axis=1)
			st.dataframe(dop, width=2160, height=50)

# Rotary Dryer calculation
with tab2:
	st.header("Rotary Dryer Section")
	col1, col2, col3 = st.columns([4,3,1])
	with col1:
		st.write("\n")
		st.write("\n")
		st.write("Insert target water content (%)")
	with col2:
		H2O_rd = st.number_input("%", min_value=0.00)
	with col3:
		st.write("\n")
		st.write("\n")
		calc = st.checkbox("Calculate!")
	
	if calc is True:
		col1, col2 = st.columns(2)
		with col1:
			st.header("Initial ore weight composition (ton)")
			st.dataframe(wow, width=2160, height=50)
		with col2:
			st.header("Ore weight composition (ton) after drying")
			global ore_rd
			ore_rd = wow.copy()
			ore_rd["H2O"] = wet_ore * H2O_rd / 100
			ore_rd.drop("Total", axis=1, inplace=True)
			ore_rd["Total"] = ore_rd.sum(axis=1)
			st.dataframe(ore_rd, width=2160, height=50)
		
		# Fuel requirement

		st.header("Fuel requirements in Rotary Dryer")

		# Inserting fuel price
		st.write("Insert current fuel price:")
		col1, col2 = st.columns(2)
		with col1:
			global hfo_price
			hfo_price = st.number_input("Heavy Fuel Oil price (USD/tonne)")
		with col2:
			global pc_price
			pc_price = st.number_input("Pulverized Coal price (USD/tonne)")

		# Insert fuel type
		st.write("\n")
		fuel = st.selectbox("Select fuel type:", ["Heavy Fuel Oil", "Pulverized Coal", "Both"])

		# Calculating amount water to be removed and its energy requirement
		r_water = (wop.loc[0, "H2O"] - H2O_rd) * wet_ore / 100
		energy_rd = r_water * 1000 * 2559.83 / 1000

		# Calculating fuel requirement
		if fuel == "Heavy Fuel Oil":
			# Net calorific value of HFO is 39.00 MJ/kg
			hfo_weight = round((energy_rd / 39.00 / 1000), 4)
			hfo_cost = round((hfo_price * hfo_weight), 4)
			col1, col2, col3, col4, col5, col6 = st.columns(6)
			with col2:
				st.metric("Water to remove", value="{} tonne".format(round(r_water, 4)))
			with col3:
				st.metric("Energy required", value="{} MJ".format(round(energy_rd, 4)))
			with col4:
				st.metric("Required HFO", value="{} tonne".format(round(hfo_weight, 4)))
			with col5:
				st.metric("Cost", value="{} USD".format(round(hfo_cost,4)))

		elif fuel == "Pulverized Coal":
			# Net calorific value of PC is 27.49 MJ/kg
			pc_weight = round((energy_rd / 27.49 / 1000), 4)
			pc_cost = round((pc_price * pc_weight), 4)
			col1, col2, col3, col4, col5, col6 = st.columns(6)
			with col2:
				st.metric("Water to remove", value="{} tonne".format(round(r_water, 4)))
			with col3:
				st.metric("Energy required", value="{} MJ".format(round(energy_rd, 4)))
			with col4:
				st.metric("Required PC", value="{} tonne".format(round(pc_weight,4)))
			with col5:
				st.metric("Cost", value="{} USD".format(round(pc_cost, 4)))

		else:
			# Create solver to find fuel composition that result in lower cost
			prob = LpProblem("Lower cost", LpMinimize)
			# Initialize variable
			x1 = LpVariable("hfo weight", 0, None)
			x2 = LpVariable("pc weight", 0, None)
			# Initialize the problem
			prob += hfo_price * x1 + pc_price * x2, "Total Cost"
			# Defining the constraint
			prob += (39.00 * x1 + 27.49 * x2) * 1000 == energy_rd, "Amount energy needed"
			prob += x1 >= 0.0001, "HFO should greater than 0"
			prob += x2 >= 0.0001, "PC should greater than 0"
			# Solve the problem
			prob.solve()
			# Show the results
			total_cost = round((x1.value() * hfo_price + x2.value() * pc_price), 4)
			st.write("Heavy Fuel Oil and Pulverized Coal requirement is {} and {} tonne, respectively with cost of {} USD". format(round(x1.value(),4), round(x2.value(),4), total_cost))
			col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
			with col2:
				st.metric("Water to remove", value="{} tonne".format(round(r_water, 4)))
			with col3:
				st.metric("Energy required", value="{} MJ".format(round(energy_rd, 4)))
			with col4:
				st.metric("Required HFO", value="{} tonne".format(round(x1.value(), 4)))
			with col5:
				st.metric("Required PC", value="{} tonne".format(round(x2.value(), 4)))				
			with col6:
				st.metric("Cost", value="{} USD".format(round(total_cost, 4)))
		
		col3, col4, col5 = st.columns(3)
		with col4:
			data_pie = ore_rd.copy()
			data_pie.drop("Total", axis=1, inplace=True)
			data_pie = data_pie.transpose().reset_index()
			data_pie.columns = ["compounds", "weight"]
			st.plotly_chart(pie_chart(data_pie, "weight", "compounds", "compounds", "Ore Composition (%)"))
	else:
		st.write("Calculation is not performed")

# Calculation for Rotary Kiln
with tab3:
	st.header("Rotary Kiln Section")
	col1, col2 = st.columns(2)
	with col1:
		# Input percentage of reduction and desired carbon percentage in calcine
		red_nio = st.number_input("Percentage of reduced NiO", min_value=0.00)
		red_fe2o3 = st.number_input("Percentage of reduced Fe2O3", min_value=0.00)
		red_feo = st.number_input("Percentage of reduced FeO", min_value=0.00)
		calc_c = st.number_input("Percentage of Carbon in calcine", min_value=0.00)
	with col2:
		# Display input and output from RK
		st.header("Input ore from RK (ton)")
		st.dataframe(ore_rd)
		st.write("\n")
		st.header("Output ore from RK (ton)")
		global ore_rk
		ore_rk = ore_rd.copy()
		ore_rk.drop(["H2O", "Total", "LOI"], axis=1, inplace=True)
		ore_rk.loc[0, "NiO"] = ore_rd.loc[0, "NiO"] * (100 - red_nio) / 100
		ore_rk.loc[0, "Fe2O3"] = ore_rd.loc[0, "Fe2O3"] * (100 - red_fe2o3) / 100
		ore_rk.loc[0, "FeO_t"] = (ore_rd.loc[0, "Fe2O3"] * red_fe2o3 / 100) * 2 * mr.loc[0, "FeO"] / mr.loc[0, "Fe2O3"]
		ore_rk.loc[0, "FeO"] = ore_rk.loc[0, "FeO_t"] * (100 - red_feo) / 100
		ore_rk.loc[0, "Ni"] = (ore_rd.loc[0, "NiO"] * red_nio / 100) * ar.loc[0, "Ni"] / mr.loc[0, "NiO"]
		ore_rk.loc[0,"Fe"] = (ore_rk.loc[0, "FeO"] * red_feo / 100) * ar.loc[0, "Fe"] / mr.loc[0, "FeO"]
		ore_rk.drop("FeO_t", axis=1, inplace=True)

		# Calculating for carbon in calcine
		perc_c = calc_c / 100 # Percentage of C in calcine
		total = ore_rk.sum(axis=1)
		ore_rk["C"] = (perc_c * total) / (1 - perc_c)
		ore_rk["Total"] = ore_rk.sum(axis=1)

		st.dataframe(ore_rk)

	# Show coal requirement for drying and reduction
	st.write("\n")
	st.write("\n")

	col1, col2 = st.columns(2)
	with col1:
		# Calculate coal requirement for drying
		st.header("Coal requirement for drying")
		total_water = ore_rd.loc[0, "H2O"] + ore_rd.loc[0, "LOI"]
		energy_rk = total_water * 1000 * 2559.83 / 1000
		d_coal = round((energy_rk / 27.49 / 1000), 4)
		st.write("Energy needed to fully dry the ore is {} MJ with coal requirement {} tonne".format(round(energy_rk, 4), d_coal))

		st.header("Reductant coal calculation")
		# Coal composition
		st.write("Insert coal composition:")
		global fc # fix carbon
		fc = st.number_input("Fix Carbon (%): ", min_value=0.00)
		global vm # volatile matter
		vm = st.number_input("Volatile Matter (%): ", min_value=0.00)
		global im # inherent moisture
		im = st.number_input("Inherent Moisture (%): ", min_value=0.00)
		global ash 
		ash = st.number_input("Ash Content (%): ", min_value=0.00)
		# Coal price
		st.write("\n")
		st.write("Insert coal price:")
		red_coal_price = st.number_input("Coal Price (USD): ", min_value=0.00)		

	# Reductant coal calculation
	with col2:
		# Calculate coal requirement for reduction
		st.header("Coal required for reduction")

		# Calculation for NiO
		st.write("For NiO reduction:")
		st.latex("NiO + C \longrightarrow Ni + CO")
		coal_nio = (ore_rd.loc[0, "NiO"] * red_nio / 100) * ar.loc[0, "C"] / mr.loc[0, "NiO"]
		co_nio = (ore_rd.loc[0, "NiO"] * red_nio / 100) * mr.loc[0, "CO"] / mr.loc[0, "NiO"]
		st.write("NiO reduction produced {} tonne CO with required C {} tonne".format(round(co_nio, 2), round(coal_nio, 2)))
		st.write("\n")
		st.write("\n")

		# Calculation for Fe2O3
		st.write("For Fe2O3 reduction:")
		st.latex("Fe_2O_3 + CO \longrightarrow 2FeO + CO_2")
		co2_fe2o3 = (ore_rd.loc[0, "Fe2O3"] * red_fe2o3 / 100) * mr.loc[0, "CO2"] / mr.loc[0, "Fe2O3"]
		co_fe2o3 = (ore_rd.loc[0, "Fe2O3"] * red_fe2o3 / 100) * mr.loc[0, "CO"] / mr.loc[0, "Fe2O3"]
		feo = (ore_rd.loc[0, "Fe2O3"] * red_fe2o3 / 100) * 2* mr.loc[0, "FeO"] / mr.loc[0, "Fe2O3"]
		st.write("Fe2O3 reduction produced {} tonne CO2 with required CO {} tonne".format(round(co2_fe2o3, 2), round(co_fe2o3, 2)))
		st.write("\n")
		st.write("\n")

		# Calculation for FeO
		st.write("For FeO reduction:")
		st.latex("FeO + C \longrightarrow Fe + CO")
		coal_feo = feo * ar.loc[0, "C"] / mr.loc[0, "FeO"]
		co_feo = feo * mr.loc[0, "CO"] / mr.loc[0, "FeO"]
		st.write("FeO reduction produced {} tonne CO with required C {} tonne".format(round(co_feo, 2), round(coal_feo, 2)))
		st.write("\n")
		st.write("\n")

		# Calculation for coal combustion to produce required CO
		st.write("For coal combustion:")
		st.latex("2C + O_2 \longrightarrow 2CO")
		required_co = co_fe2o3 - co_nio
		required_carbon = required_co * 2 * ar.loc[0, "C"] / (2 * mr.loc[0, "CO"])
		st.write("To fulfill the required CO for Fe2O3 reduction, {} tonne of carbon is needed".format(round(required_carbon, 2)))
	
	# PC calculation and reductant coal calculation with its cost
	st.write("\n")
	st.write("\n")

	pc_drying_cost = d_coal * pc_price
	red_coal_weight = (required_carbon / (fc / 100)) + ore_rk.loc[0, "C"]
	red_coal_cost = red_coal_weight * red_coal_price

	col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
	with col2:
		st.metric("Pulverized Coal required", value="{} tonne".format(d_coal))
	with col3:
		st.metric("Pulverized Coal cost", value="{} USD".format(round(pc_price, 4)))
	with col5:
		st.metric("Reductant Coal required", value="{} tonne".format(round(red_coal_weight, 4)))
	with col6:
		st.metric("Reductant Coal cost", value="{} USD".format(round(red_coal_cost, 4)))	

	# Show pie chart about calcine composition
	col1, col2, col3 = st.columns(3)
	with col2:
		data_pie = ore_rk.copy()
		data_pie.drop("Total", axis=1, inplace=True)
		data_pie = data_pie.transpose().reset_index()
		data_pie.columns = ["compounds", "weight"]
		st.plotly_chart(pie_chart(data_pie, "weight", "compounds", "compounds", "Calcine Composition (%)"))
