import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

# IMPORT
# OK = Ordinary Kriging 
from pyvista_new import Gempy as Gempy_OK          

# UK = Universal Kriging (Polynomial basis)
from withbasisfunction import Gempy as Gempy_UK    


class GempyFaultModel(Gempy_UK):
    """
    Extension of 4D Gempy class to handle faults using a Dual-Engine architecture.
    
    
    
    Extension of 4D Gempy class to handle faults.
    - Fault Engine: Uses Ordinary Kriging for strict, flat boundaries.
    Block-wise Solving.

    - Structure: Uses basis/Universal Kriging for natural structural folding.
    Splitting the structural dataset based on the fault surface 
    and interpolate both sides independently.
    """

    def __init__(self, project_name, extent, resolution):

        ## calling initialization of the parent 'Gempy' class for structural model
        ## Using universal kriging UK
        super().__init__(project_name, extent, resolution)
        
        # Initialize a completely separate Fault Engine (Ordinary Kriging) from pyvista_new.py
        self.fault_engine = Gempy_OK(project_name + "_Fault", extent, resolution)
        
        # Storing independent kriging systems/weights (fault, hanging wall(hw), footwall(fw))
        self.hw_state = None  # Hanging wall (Zone 1)
        self.fw_state = None  # Footwall (Zone 2)

    
        # Stores scalar value that defines where the fault plane is
        self.fault_threshold = 0.0 

    def save_internal_state(self):
        """Saves all parameters to avoid overwriting between
        different systems.
        """
        
        state = {
            'w': self.w.clone(),
            'Transformation_matrix': self.Transformation_matrix.clone(), 
            'Position_G': self.Position_G.clone(),
            'Value_G': self.Value_G.clone(),  
            'Position_G_Modified': self.Position_G_Modified.clone(),
            'ref_layer_points': self.ref_layer_points.clone(),
            'rest_layer_points': self.rest_layer_points.clone(),
            'number_of_layer': self.number_of_layer,
            'number_of_points_per_surface': self.number_of_points_per_surface.clone(),
            'sp_coord': copy.deepcopy(self.sp_coord)
        }
        return state

    def load_internal_state(self, state):
        """
         We can load specific kriging system paratmeters.
        """
        self.w = state['w']
        self.Transformation_matrix = state['Transformation_matrix'] 
        self.Position_G = state['Position_G']
        self.Value_G = state['Value_G'] 
        self.Position_G_Modified = state['Position_G_Modified']
        self.ref_layer_points = state['ref_layer_points']
        self.rest_layer_points = state['rest_layer_points']
        self.number_of_layer = state['number_of_layer']
        self.number_of_points_per_surface = state['number_of_points_per_surface']
        self.sp_coord = state['sp_coord']


    def compute_models(self, fault_data, structure_data):
        '''
        Computes Fault Model ->

        Evaluates Fault field at Structural Data coordinates ->

        Partitions/Masks data into Hanging Wall and Footwall arrays by using Fautl weights ->

        Interpolates HW block and FW block independently.
        '''
        
        ######### Solving Fault Model #########
        ###### USING Ordinary kriging
        print("\n Solving Initial Fault Kriging System (Ordinary Kriging)")

        self.fault_engine.interface_data(fault_data['sp_coord'])
        self.fault_engine.orientation_data(fault_data['op_coord'])
        self.fault_engine.Transformation_matrix = fault_data.get('transformation_matrix', torch.diag(torch.tensor([1.0,1.0,1.0,0.0])))

        # Solves Fault model   
        self.fault_engine.Ge_model() 

        # Calculate fault threshold using the fault engine
        # We must explicitly pass the fault engine's ref_layer_points
        fault_out, _ = self.fault_engine.Solution_grid(self.fault_engine.ref_layer_points, section_plot=True, recompute_weights=False)
        if "scalar_ref_points" in fault_out:
            self.fault_threshold = torch.mean(fault_out["scalar_ref_points"])

        ######### Partitioning Data into footwall and hanging wall #########
        print("\n Partitioning Structural Data (Zone 1 vs Zone 2)")
        
        struct_matrix = structure_data.get('transformation_matrix', torch.diag(torch.tensor([1.0,1.0,1.0,0.0])))
        
        hw_struct_data = {'sp_coord': {}, 'op_coord': {'Positions': [], 'Values': []}, 'transformation_matrix': struct_matrix}
        fw_struct_data = {'sp_coord': {}, 'op_coord': {'Positions': [], 'Values': []}, 'transformation_matrix': struct_matrix}
        
        # Splitting Interface Points
        for layer_name, points in structure_data['sp_coord'].items():
            # Evaluate using the FAULT ENGINE
            fault_scalars, _ = self.fault_engine.Solution_grid(points, section_plot=True, recompute_weights=False)
            scalars = fault_scalars["Regular"]
            
            hw_mask = (scalars > self.fault_threshold).squeeze()
            fw_mask = ~hw_mask
            
            if hw_mask.sum() > 0: hw_struct_data['sp_coord'][layer_name] = points[hw_mask]
            if fw_mask.sum() > 0: fw_struct_data['sp_coord'][layer_name] = points[fw_mask]
                
        # Splitting Orientation Points
        op_pos = structure_data['op_coord']['Positions']
        op_val = structure_data['op_coord']['Values']
        
        fault_scalars, _ = self.fault_engine.Solution_grid(op_pos, section_plot=True, recompute_weights=False)
        scalars = fault_scalars["Regular"]
        
        hw_mask = (scalars > self.fault_threshold).squeeze()
        fw_mask = ~hw_mask
        
        if hw_mask.sum() > 0:
            hw_struct_data['op_coord']['Positions'] = op_pos[hw_mask]
            hw_struct_data['op_coord']['Values'] = op_val[hw_mask]
        if fw_mask.sum() > 0:
            fw_struct_data['op_coord']['Positions'] = op_pos[fw_mask]
            fw_struct_data['op_coord']['Values'] = op_val[fw_mask]

        #########  Solving Zone 1: Hanging Wall (Using Curvy Engine) #########
        self.hw_state = None
        if len(hw_struct_data['op_coord']['Positions']) > 0 and len(hw_struct_data['sp_coord']) > 0:
            print("\n- Solving Hanging Wall (Zone 1) Structure (Universal Kriging) -")
            self.interface_data(hw_struct_data['sp_coord'])
            self.orientation_data(hw_struct_data['op_coord'])
            self.Transformation_matrix = hw_struct_data['transformation_matrix']
            
            self.Ge_model() # This is self, so it uses the Universal Kriging engine
            self.hw_state = self.save_internal_state()
        else:
            print("\nWarning: Not enough data for Hanging Wall interpolation.")

        ######### Solving Zone 2: Footwall (Using Curvy Engine) #########
        self.fw_state = None
        if len(fw_struct_data['op_coord']['Positions']) > 0 and len(fw_struct_data['sp_coord']) > 0:
            print("\n- Solving Footwall (Zone 2) Structure (Universal Kriging) -")
            self.interface_data(fw_struct_data['sp_coord'])
            self.orientation_data(fw_struct_data['op_coord'])
            self.Transformation_matrix = fw_struct_data['transformation_matrix']
            
            self.Ge_model()
            self.fw_state = self.save_internal_state()
        else:
             print("\nWarning: Not enough data for Footwall interpolation.")

        ######### Combine input data for plotting ########
        print("\n--- Combining Original Data for Visualization ---")
        self.sp_coord = {**fault_data['sp_coord'], **structure_data['sp_coord']}
        self.Position_G = torch.cat([fault_data['op_coord']['Positions'], structure_data['op_coord']['Positions']], dim=0)
        self.Value_G = torch.cat([fault_data['op_coord']['Values'], structure_data['op_coord']['Values']], dim=0)
        
        # A dummy weight tensor to trick the parent visualization function
        self.w = torch.zeros(1) 

        print("Model computation successfully for each zone.")

    def compute_faulted_grid(self, grid_coord=None):
        """
        Evaluates the HW and FW models over the entire 3D grid and masks
        them together based on the fault block geometry.
        """
        if grid_coord is None:
            grid_coord = self.data["Regular"]

        # BACKUP combined state
        combined_backup = self.save_internal_state()

        try:
            # 1. Evaluate Fault Block Mask for whole grid using the FAULT ENGINE
            fault_out, _ = self.fault_engine.Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
            fault_scalars = fault_out["Regular"]
            hw_mask = fault_scalars > self.fault_threshold
            
            struct_out_hw, struct_res_hw = None, None
            struct_out_fw, struct_res_fw = None, None

            # 2. Evaluate Hanging Wall Structural Grid using self (UK Engine)
            if self.hw_state is not None:
                self.load_internal_state(self.hw_state)
                struct_out_hw, struct_res_hw = super().Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
                
            # 3. Evaluate Footwall Structural Grid using self (UK Engine)
            if self.fw_state is not None:
                self.load_internal_state(self.fw_state)
                struct_out_fw, struct_res_fw = super().Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
                
            # 4. Stitch Outputs based on Fault Mask
            final_out = {}
            final_res = {}
            
            if struct_out_hw is not None and struct_out_fw is not None:
                for k in struct_out_hw.keys():
                    if k == 'scalar_ref_points':
                        final_out[k] = struct_out_hw[k]
                    else:
                        final_out[k] = torch.where(hw_mask, struct_out_hw[k], struct_out_fw[k])
                        
                for k in struct_res_hw.keys():
                    if k == 'ref_points':
                        final_res[k] = struct_res_hw[k]
                    else:
                        final_res[k] = torch.where(hw_mask, struct_res_hw[k], struct_res_fw[k])

            # Fallbacks if a zone failed
            elif struct_out_hw is not None:
                final_out, final_res = struct_out_hw, struct_res_hw
            elif struct_out_fw is not None:
                final_out, final_res = struct_out_fw, struct_res_fw
                
            return final_out, final_res

        finally:
            # RESTORE combined state for plot tools
            self.load_internal_state(combined_backup)

    def Solution_grid(self, grid_coord=None, section_plot=False, recompute_weights=True):
        """Overrides parent method so plotting functions automatically use stitched grid."""
        return self.compute_faulted_grid(grid_coord)


if __name__ == "__main__":

    # Define Gempy model paremters like extent resolution
    extent = [-0.1, 1.1, 0.1, 0.9, 0.1, 0.90, 0, 5]
    resolution = [100, 50, 50, 2]

    model = GempyFaultModel("Block Model Test", extent, resolution)
    
    # 1. FAULT Data
    fault_interface_data = {
    'fault': torch.tensor([
        [500.0, 500.0, 500.0, 0.0],
        [450.0, 500.0, 600.0, 0.0],
        [500.0, 200.0, 500.0, 0.0],
        [450.0, 200.0, 600.0, 0.0],
        [500.0, 800.0, 500.0, 0.0],
        [450.0, 800.0, 600.0, 0.0],

        # NEW POINTS
        [500.0, 800.0, 510.0, 0.0],
        [450.0, 800.0, 590.0, 0.0]
        ])/1000
    }
    
    fault_orientation_data = {
        'Positions': torch.tensor([ [500.0, 500.0, 500.0, 0] ]) / 1000,
        "Values": torch.tensor([[0.894, 0, 0.447, 0]])
    }
    
    fault_transformation_matrix = torch.diag(torch.tensor([1.0, 1.0, 1.0, 0.05]))

    fault_input = {'sp_coord': fault_interface_data,
                    'op_coord': fault_orientation_data,
                    'transformation_matrix': fault_transformation_matrix}

    # 2. STRUCTURE Data
    struct_interface_data =  {
        "rock1": torch.tensor([
        [0.0, 200.0, 600.0, 0.0], [0.0, 500.0, 600.0, 0.0], [0.0, 800.0, 600.0, 0.0],
        [200.0, 200.0, 600.0, 0.0], [200.0, 500.0, 600.0, 0.0], [200.0, 800.0, 600.0, 0.0],
        [800.0, 200.0, 200.0, 0.0], [800.0, 500.0, 200.0, 0.0], [800.0, 800.0, 200.0, 0.0],
        [1000.0, 200.0, 200.0, 0.0], [1000.0, 500.0, 200.0, 0.0], [1000.0, 800.0, 200.0, 0.0],
    ]) / 1000,

    "rock2": torch.tensor([
        [0.0, 200.0, 800.0, 0.0], [0.0, 800.0, 800.0, 0.0], [200.0, 200.0, 800.0, 0.0],
        [200.0, 800.0, 800.0, 0.0], [800.0, 200.0, 400.0, 0.0], [800.0, 800.0, 400.0, 0.0],
        [1000.0, 200.0, 400.0, 0.0], [1000.0, 800.0, 400.0, 0.0],
    ]) / 1000
    }
    
    struct_orientation_data = {
       "Positions": torch.tensor([
        [100.0, 500.0, 800.0, 0.0], [100.0, 500.0, 600.0, 0.0],
        [900.0, 500.0, 400.0, 0.0], [900.0, 500.0, 200.0, 0.0],
    ]) / 1000,
    "Values": torch.tensor([
        [0.000, 0.000, 1.000, 0.1], [0.000, 0.000, 1.000, 0.1],
        [0.000, 0.000, 1.000, -0.1], [0.000, 0.000, 1.000, -0.1],
    ])
    }

    struct_transformation_matrix = torch.diag(torch.tensor([1.0, 1.0, 1.0, 0.05]))


    struct_input = {'sp_coord': struct_interface_data,
                     'op_coord': struct_orientation_data,
                     'transformation_matrix': struct_transformation_matrix}

    # Solving
    model.compute_models(fault_data=fault_input, structure_data=struct_input)
    print("\nModel computed successfully.")

    # --- PLOTTING ---
    #########################################################################
    ###### Uncomment the below code lines for matplotlib visualization ######
    #########################################################################

    ##### FOR 2D matplotlib #####
    import time
    for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3]:
        model.plot_data_section(section={2:0.5, 4:t}, plot_scalar_field = True, plot_input_data=True)
        time.sleep(1)


    ##### FOR 3D matplotlib #####
    # import time
    # for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 3.5, 4,4.5]:
    #     model.plot_data_section(section={4:t}, plot_scalar_field = True, plot_input_data=True)
    #     time.sleep(1)/

    #############################################################################################
    ########## Uncomment below for  Interactive Visualization Pyvista below  ####################
    #############################################################################################
    
    ###############################################################
    ########### show/unshow input data using "plot_input_data" argument
    ########### show/unshow surface or interfaces using "only_surface_mode" argument
    ###############################################################

    # 

    # --- PLOTTING ---
    # model.plot_interactive_section(plot_input_data=True, only_surface_mode= False)