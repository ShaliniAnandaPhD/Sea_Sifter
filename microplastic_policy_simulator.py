import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class MicroplasticPolicySimulator:
    def __init__(self, plastic_production, plastic_consumption, waste_management, policies):
        """
        Initialize the Microplastic Policy Simulator.

        :param plastic_production: DataFrame containing plastic production data
        :param plastic_consumption: DataFrame containing plastic consumption data
        :param waste_management: DataFrame containing waste management data
        :param policies: DataFrame containing environmental policies data
        """
        self.plastic_production = plastic_production
        self.plastic_consumption = plastic_consumption
        self.waste_management = waste_management
        self.policies = policies

    def preprocess_data(self):
        """
        Preprocess the input data.

        Possible errors:
        - Missing or inconsistent data in the input DataFrames
        - Incompatible data formats or data types

        Solutions:
        - Handle missing data by filling or removing rows/columns as appropriate
        - Convert data types and formats to ensure consistency
        """
        try:
            # Handle missing data
            self.plastic_production.fillna(0, inplace=True)
            self.plastic_consumption.fillna(0, inplace=True)
            self.waste_management.fillna(0, inplace=True)
            self.policies.fillna(0, inplace=True)

            # Convert data types and formats
            self.plastic_production = self.plastic_production.astype(float)
            self.plastic_consumption = self.plastic_consumption.astype(float)
            self.waste_management = self.waste_management.astype(float)
            self.policies = self.policies.astype(int)

        except (ValueError, TypeError) as e:
            print(f"Error occurred during data preprocessing: {str(e)}")
            print("Please check the input data and ensure it is in the correct format.")

    def simulate_scenario(self, scenario, time_steps):
        """
        Simulate a specific scenario.

        :param scenario: Dictionary containing scenario parameters
        :param time_steps: Number of time steps to simulate

        Possible errors:
        - Invalid scenario parameters
        - Insufficient time steps for accurate simulation

        Solutions:
        - Validate scenario parameters before running the simulation
        - Adjust the number of time steps based on the complexity of the scenario
        """
        try:
            # Validate scenario parameters
            required_params = ['ban_single_use', 'improve_waste_mgmt', 'extended_producer_resp']
            for param in required_params:
                if param not in scenario:
                    raise ValueError(f"Missing scenario parameter: {param}")

            # Set up initial conditions
            initial_microplastic = scenario.get('initial_microplastic', 0)
            initial_conditions = [initial_microplastic]

            # Define the time points for the simulation
            time_points = np.linspace(0, time_steps, time_steps + 1)

            # Run the simulation using the ODEs
            results = odeint(self.microplastic_odes, initial_conditions, time_points, args=(scenario,))

            # Extract the microplastic levels from the simulation results
            microplastic_levels = results[:, 0]

            return microplastic_levels

        except ValueError as ve:
            print(f"Invalid scenario parameter: {str(ve)}")
            print("Please provide all the required scenario parameters.")

        except Exception as e:
            print(f"An error occurred during scenario simulation: {str(e)}")
            print("Please check the scenario parameters and try again.")

    def microplastic_odes(self, microplastic, t, scenario):
        """
        Define the ordinary differential equations (ODEs) for microplastic dynamics.

        :param microplastic: Current microplastic level
        :param t: Current time step
        :param scenario: Dictionary containing scenario parameters

        Possible errors:
        - Division by zero if certain parameters are not properly defined
        - Incorrect ODE formulation leading to unrealistic results

        Solutions:
        - Handle division by zero by checking for zero values and providing appropriate defaults
        - Validate the ODE formulation and ensure it aligns with the underlying microplastic dynamics
        """
        try:
            # Extract scenario parameters
            ban_single_use = scenario['ban_single_use']
            improve_waste_mgmt = scenario['improve_waste_mgmt']
            extended_producer_resp = scenario['extended_producer_resp']

            # Calculate the rates based on scenario parameters
            production_rate = self.get_production_rate(t)
            consumption_rate = self.get_consumption_rate(t)
            waste_mgmt_rate = self.get_waste_mgmt_rate(t)

            # Apply scenario-specific modifications to the rates
            if ban_single_use:
                consumption_rate *= 0.8  # Reduce consumption rate by 20%
            if improve_waste_mgmt:
                waste_mgmt_rate *= 1.2  # Increase waste management rate by 20%
            if extended_producer_resp:
                production_rate *= 0.9  # Reduce production rate by 10%

            # Define the ODEs for microplastic dynamics
            d_microplastic_dt = production_rate + consumption_rate - waste_mgmt_rate

            return [d_microplastic_dt]

        except ZeroDivisionError:
            print("Division by zero encountered in the ODEs.")
            print("Please check the parameter values and ensure they are properly defined.")
            return [0]

        except Exception as e:
            print(f"An error occurred in the microplastic ODEs: {str(e)}")
            print("Please validate the ODE formulation and parameter values.")
            return [0]

    def get_production_rate(self, t):
        """
        Calculate the plastic production rate at a given time step.

        :param t: Current time step

        Possible errors:
        - Missing or invalid data in the plastic_production DataFrame
        - Incorrect indexing or data retrieval

        Solutions:
        - Handle missing data by providing default values or interpolation
        - Ensure correct indexing and data retrieval from the DataFrame
        """
        try:
            production_rate = self.plastic_production.loc[t, 'production_rate']
            return production_rate

        except KeyError:
            print(f"Missing production data for time step {t}.")
            print("Using default production rate of 0.")
            return 0

        except Exception as e:
            print(f"An error occurred while retrieving the production rate: {str(e)}")
            print("Please check the plastic_production DataFrame and ensure it is properly formatted.")
            return 0

    def get_consumption_rate(self, t):
        """
        Calculate the plastic consumption rate at a given time step.

        :param t: Current time step

        Possible errors:
        - Missing or invalid data in the plastic_consumption DataFrame
        - Incorrect indexing or data retrieval

        Solutions:
        - Handle missing data by providing default values or interpolation
        - Ensure correct indexing and data retrieval from the DataFrame
        """
        try:
            consumption_rate = self.plastic_consumption.loc[t, 'consumption_rate']
            return consumption_rate

        except KeyError:
            print(f"Missing consumption data for time step {t}.")
            print("Using default consumption rate of 0.")
            return 0

        except Exception as e:
            print(f"An error occurred while retrieving the consumption rate: {str(e)}")
            print("Please check the plastic_consumption DataFrame and ensure it is properly formatted.")
            return 0

    def get_waste_mgmt_rate(self, t):
        """
        Calculate the waste management rate at a given time step.

        :param t: Current time step

        Possible errors:
        - Missing or invalid data in the waste_management DataFrame
        - Incorrect indexing or data retrieval

        Solutions:
        - Handle missing data by providing default values or interpolation
        - Ensure correct indexing and data retrieval from the DataFrame
        """
        try:
            waste_mgmt_rate = self.waste_management.loc[t, 'waste_mgmt_rate']
            return waste_mgmt_rate

        except KeyError:
            print(f"Missing waste management data for time step {t}.")
            print("Using default waste management rate of 0.")
            return 0

        except Exception as e:
            print(f"An error occurred while retrieving the waste management rate: {str(e)}")
            print("Please check the waste_management DataFrame and ensure it is properly formatted.")
            return 0

    def plot_results(self, results, title):
        """
        Plot the simulation results.

        :param results: Dictionary containing simulation results for each scenario
        :param title: Title of the plot

        Possible errors:
        - Missing or invalid data in the results dictionary
        - Incorrect plotting configuration or labels

        Solutions:
        - Validate the structure and contents of the results dictionary
        - Ensure proper plotting configuration, labels, and legend
        """
        try:
            plt.figure(figsize=(10, 6))
            for scenario, microplastic_levels in results.items():
                plt.plot(microplastic_levels, label=scenario)
            plt.xlabel('Time Steps')
            plt.ylabel('Microplastic Levels')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.show()

        except KeyError as ke:
            print(f"Missing scenario data in the results dictionary: {str(ke)}")
            print("Please ensure all scenarios are properly simulated and included in the results.")

        except Exception as e:
            print(f"An error occurred while plotting the results: {str(e)}")
            print("Please check the plotting configuration and input data.")

def main():
    """
    Main function to run the Microplastic Policy Simulator.
    """
    # Load input data (replace with actual data loading code)
    plastic_production = pd.DataFrame({'production_rate': [10, 12, 15, 18, 20]})
    plastic_consumption = pd.DataFrame({'consumption_rate': [8, 10, 12, 15, 18]})
    waste_management = pd.DataFrame({'waste_mgmt_rate': [5, 6, 7, 8, 9]})
    policies = pd.DataFrame({'ban_single_use': [0, 0, 1, 1, 1],
                             'improve_waste_mgmt': [0, 1, 1, 1, 1],
                             'extended_producer_resp': [0, 0, 0, 1, 1]})

    # Create an instance of the MicroplasticPolicySimulator
    simulator = MicroplasticPolicySimulator(plastic_production, plastic_consumption, waste_management, policies)

    # Preprocess the input data
    simulator.preprocess_data()

    # Define scenarios
    scenarios = {
        'Business as usual': {'ban_single_use': 0, 'improve_waste_mgmt': 0, 'extended_producer_resp': 0},
        'Ban single-use plastics': {'ban_single_use': 1, 'improve_waste_mgmt': 0, 'extended_producer_resp': 0},
        'Improve waste management': {'ban_single_use': 0, 'improve_waste_mgmt': 1, 'extended_producer_resp': 0},
        'Extended producer responsibility': {'ban_single_use': 0, 'improve_waste_mgmt': 0, 'extended_producer_resp': 1},
        'Combined policies': {'ban_single_use': 1, 'improve_waste_mgmt': 1, 'extended_producer_resp': 1}
    }

    # Simulate scenarios
    results = {}
    for scenario_name, scenario_params in scenarios.items():
        microplastic_levels = simulator.simulate_scenario(scenario_params, time_steps=5)
        results[scenario_name] = microplastic_levels

    # Plot the simulation results
    simulator.plot_results(results, title='Microplastic Policy Simulation Results')

if __name__ == '__main__':
    main()
