"""
Thin CoolProp wrapper for the water-side property lookups in Module 3 and
the exergy functions.
"""

import CoolProp.CoolProp as CP


class ThermoCore:
    def __init__(self, T0_celsius=25.0, P0_kpa=101.325):
        """
        Initializes the thermodynamic core and sets the absolute dead state
        based on local ambient atmospheric design conditions.
        """
        self.T0 = T0_celsius + 273.15  # Convert to Kelvin
        self.P0 = P0_kpa * 1000.0  # Convert kPa to Pa

        # Fluids used across the 3 architectures
        self.WATER = "Water"
        self.ORC_BASE = "R245fa"
        self.ORC_TRANS = "R1234ze(E)"

        # Pre-calculate dead-state properties for flow exergy baselines
        self.h0_water = CP.PropsSI("H", "T", self.T0, "P", self.P0, self.WATER)
        self.s0_water = CP.PropsSI("S", "T", self.T0, "P", self.P0, self.WATER)

        self.h0_r245fa = CP.PropsSI("H", "T", self.T0, "P", self.P0, self.ORC_BASE)
        self.s0_r245fa = CP.PropsSI("S", "T", self.T0, "P", self.P0, self.ORC_BASE)

    def get_fluid_properties(self, fluid_name, input1_type, input1_val, input2_type, input2_val):
        """
        Thin wrapper for CoolProp PropsSI to retrieve state variables safely.
        Inputs use standard CoolProp characters (e.g., 'T' for Temp in K, 'P' for Pressure in Pa).
        Returns a dict of Enthalpy (J/kg), Entropy (J/kg-K), and Quality (Q).
        """
        try:
            h = CP.PropsSI("H", input1_type, input1_val, input2_type, input2_val, fluid_name)
            s = CP.PropsSI("S", input1_type, input1_val, input2_type, input2_val, fluid_name)

            # Phase check to handle single-phase vs two-phase quality lookups safely
            try:
                q = CP.PropsSI("Q", input1_type, input1_val, input2_type, input2_val, fluid_name)
            except Exception:
                q = -1.0  # Returns -1.0 if fluid is subcooled or superheated

            return {"h": h, "s": s, "q": q}
        except Exception as e:
            raise ValueError(
                f"CoolProp lookup failed for {fluid_name} with inputs ({input1_type}={input1_val}, {input2_type}={input2_val}): {e}")

    def calculate_flow_exergy(self, fluid_name, h, s):
        """
        Calculates specific flow exergy (J/kg) relative to the defined dead state.
        """
        if fluid_name == self.WATER:
            h0, s0 = self.h0_water, self.s0_water
        elif fluid_name == self.ORC_BASE:
            h0, s0 = self.h0_r245fa, self.s0_r245fa
        else:
            # Dynamically calculate for alternative fluids if swept in Module 9
            h0 = CP.PropsSI("H", "T", self.T0, "P", self.P0, fluid_name)
            s0 = CP.PropsSI("S", "T", self.T0, "P", self.P0, fluid_name)

        specific_exergy = (h - h0) - (self.T0 * (s - s0))
        return specific_exergy
