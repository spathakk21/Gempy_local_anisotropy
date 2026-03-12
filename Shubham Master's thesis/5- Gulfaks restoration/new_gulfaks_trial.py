import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import Gempy class from pyvista_new.py or withbasisfunction.py(with universal term)
from withbasisfunction import Gempy

class GempyMultiFaultModel(Gempy):
    """
    Handles multiple faults dynamically using Boolean Block Signatures.
    
    Extension of 4D Gempy class to handle faults.
    Block-wise Solving.
    Splitting the structural dataset based on the fault surface 
    and interpolate both sides independently.

    """

    def __init__(self, project_name, extent, resolution):

        ## calling initialization of the parent 'Gempy' class
        super().__init__(project_name, extent, resolution)
        
        # Storing independent kriging systems
        # Dictionaries to store N fault models and N block models
        self.fault_states = {}
        self.fault_thresholds = {}
        self.block_states = {}  # Keys will be boolean tuples e.g., (True, False)

    def save_internal_state(self):
        """Saves all parameters to avoid overwriting between
        different systems.
        """
        return {
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

    def eval_fault_at_points(self, fault_name, points):
        """Evaluates a specific fault's scalar field at given coordinates."""
        self.load_internal_state(self.fault_states[fault_name])
        out, _ = super().Solution_grid(points, section_plot=True, recompute_weights=False)
        return out["Regular"]

    def compute_models(self, faults_data, structure_data):
        '''
        faults_data: Dictionary of multiple faults e.g., {'Fault1': f1_data, 'Fault2': f2_data}
        '''
        default_matrix = torch.diag(torch.tensor([1,1,1,0]))
        self.fault_names = list(faults_data.keys())
        
        ######### Solving All Fault Models #########
        for f_name in self.fault_names:
            print(f"\n- Solving Fault: {f_name} -")

            # Choose the present fault to be evaluated
            f_data = faults_data[f_name]


            self.interface_data(f_data['sp_coord'])
            self.orientation_data(f_data['op_coord'])
            self.Transformation_matrix = f_data.get('transformation_matrix', default_matrix)
            
            self.Ge_model() 
            # Saving current fault model

            self.fault_states[f_name] = self.save_internal_state()

            # Capture current fault threshold
            out, _ = super().Solution_grid(self.ref_layer_points, section_plot=True, recompute_weights=False)
            # capturing mean of reference points of current fault
            self.fault_thresholds[f_name] = torch.mean(out["scalar_ref_points"])

        ######### Partitioning Data by Boolean Signatures #########
        print("\n- Partitioning Structural Data into Blocks -")
        blocks_struct_data = {}

        def get_signatures(points):
            """Returns a list of boolean tuples representing the block signature for each point."""
            if len(points) == 0: return []
            sigs = []
            for f_name in self.fault_names:
                scalars = self.eval_fault_at_points(f_name, points).squeeze()
                if scalars.dim() == 0: scalars = scalars.unsqueeze(0)
                mask = scalars > self.fault_thresholds[f_name]
                sigs.append(mask)
            
            # Stack and convert to rows of tuples
            sigs_stacked = torch.stack(sigs, dim=1)
            return [tuple(row.tolist()) for row in sigs_stacked]

        # Route Interface Points to correct block dictionary
        for layer_name, points in structure_data['sp_coord'].items():
            sigs = get_signatures(points)
            for i, sig in enumerate(sigs):
                if sig not in blocks_struct_data:
                    blocks_struct_data[sig] = {'sp_coord': {}, 'op_coord': {'Positions': [], 'Values': []}}
                if layer_name not in blocks_struct_data[sig]['sp_coord']:
                    blocks_struct_data[sig]['sp_coord'][layer_name] = []
                blocks_struct_data[sig]['sp_coord'][layer_name].append(points[i])

        # Route Orientation Points to correct block dictionary
        op_pos = structure_data['op_coord']['Positions']
        op_val = structure_data['op_coord']['Values']
        sigs = get_signatures(op_pos)
        
        for i, sig in enumerate(sigs):
            if sig not in blocks_struct_data:
                blocks_struct_data[sig] = {'sp_coord': {}, 'op_coord': {'Positions': [], 'Values': []}}
            blocks_struct_data[sig]['op_coord']['Positions'].append(op_pos[i])
            blocks_struct_data[sig]['op_coord']['Values'].append(op_val[i])

        # Convert grouped lists back to PyTorch tensors
        for sig in blocks_struct_data:
            for layer_name in blocks_struct_data[sig]['sp_coord']:
                blocks_struct_data[sig]['sp_coord'][layer_name] = torch.stack(blocks_struct_data[sig]['sp_coord'][layer_name])
            if len(blocks_struct_data[sig]['op_coord']['Positions']) > 0:
                blocks_struct_data[sig]['op_coord']['Positions'] = torch.stack(blocks_struct_data[sig]['op_coord']['Positions'])
                blocks_struct_data[sig]['op_coord']['Values'] = torch.stack(blocks_struct_data[sig]['op_coord']['Values'])

        ######### 3. Solve Kriging for Each Populated Block #########
        self.block_states = {}
        for sig, data in blocks_struct_data.items():
            has_interfaces = len(data['sp_coord']) > 0
            has_orientations = len(data['op_coord']['Positions']) > 0
            
            if has_interfaces and has_orientations:
                print(f"\n--- Solving Structural Block {sig} ---")
                self.interface_data(data['sp_coord'])
                self.orientation_data(data['op_coord'])
                self.Transformation_matrix = structure_data.get('transformation_matrix', default_matrix)
                self.Ge_model()
                self.block_states[sig] = self.save_internal_state()
            else:
                print(f"\nWarning: Block {sig} skipped (Needs at least 1 interface and 1 orientation point in this block).")

        ######### 4. Combine Original Data for Visualization #########
        self.sp_coord = structure_data['sp_coord'].copy()
        for f_data in faults_data.values():
            self.sp_coord.update(f_data['sp_coord'])
            
        all_op_pos = [structure_data['op_coord']['Positions']] + [f['op_coord']['Positions'] for f in faults_data.values()]
        all_op_val = [structure_data['op_coord']['Values']] + [f['op_coord']['Values'] for f in faults_data.values()]
        
        self.Position_G = torch.cat(all_op_pos, dim=0)
        self.Value_G = torch.cat(all_op_val, dim=0)
        self.w = torch.zeros(1) 

    def compute_faulted_grid(self, grid_coord=None):
        if grid_coord is None:
            grid_coord = self.data["Regular"]

        combined_backup = self.save_internal_state()

        try:
            # 1. Evaluate all fault boundaries on the 3D Grid
            grid_fault_masks = {}
            for f_name in self.fault_names:
                self.load_internal_state(self.fault_states[f_name])
                out, _ = super().Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
                grid_fault_masks[f_name] = out["Regular"] > self.fault_thresholds[f_name]

            # 2. Evaluate all populated structural blocks
            block_outputs, block_res = {}, {}
            for sig in self.block_states:
                self.load_internal_state(self.block_states[sig])
                out, res = super().Solution_grid(grid_coord, section_plot=True, recompute_weights=False)
                block_outputs[sig] = out
                block_res[sig] = res

            if not block_outputs:
                raise ValueError("No valid structural blocks were computed!")


            ##########
            ############### SCALAR FIELD NORMALIZATION
            
            # Pick the first computed block as the "reference/master" block
            template_sig = list(block_outputs.keys())[0]
            master_ref_mean = torch.mean(block_outputs[template_sig]['scalar_ref_points'])

            # print(f"\n--- Normalizing {len(block_outputs)} Blocks to Master/reference {template_sig} ---")
            
            # Iterate through ALL blocks and shift them to match the Master
            for sig, out_dict in block_outputs.items():
                if sig != template_sig: # Skip the master block itself
                    # Find this specific block's average scaa=lar field value
                    block_ref_mean = torch.mean(out_dict['scalar_ref_points'])
                    
                    # Calculate the difference in the values
                    scalar_shift = master_ref_mean - block_ref_mean
                    
                    # Apply the shift to the block's grid and reference points
                    out_dict['Regular'] = out_dict['Regular'] + scalar_shift
                    out_dict['scalar_ref_points'] = out_dict['scalar_ref_points'] + scalar_shift
                    
                    # print(f"   -> Block {sig} shifted by: {scalar_shift.item()")

            ###########

            # 3. Stitch blocks together based on Boolean Signatures
            final_out, final_res = {}, {}
            template_sig = list(block_outputs.keys())[0]
            
            # Initialize empty tensors
            for k in block_outputs[template_sig].keys():
                final_out[k] = block_outputs[template_sig][k] if k == 'scalar_ref_points' else torch.zeros_like(block_outputs[template_sig][k])
            for k in block_res[template_sig].keys():
                final_res[k] = block_res[template_sig][k] if k == 'ref_points' else torch.zeros_like(block_res[template_sig][k])

            # Apply masking
            for sig, out_dict in block_outputs.items():
                # Build compound mask for this specific block
                block_mask = torch.ones_like(grid_fault_masks[self.fault_names[0]], dtype=torch.bool)
                for i, f_name in enumerate(self.fault_names):
                    expected = sig[i]
                    if expected:
                        block_mask = block_mask & grid_fault_masks[f_name]
                    else:
                        block_mask = block_mask & (~grid_fault_masks[f_name])

                # Fill data into final output where the mask is True
                for k in out_dict.keys():
                    if k != 'scalar_ref_points':
                        final_out[k] = torch.where(block_mask, out_dict[k], final_out[k])
                for k in block_res[sig].keys():
                    if k != 'ref_points':
                        final_res[k] = torch.where(block_mask, block_res[sig][k], final_res[k])

            return final_out, final_res

        finally:
            self.load_internal_state(combined_backup)

    def Solution_grid(self, grid_coord=None, section_plot=False, recompute_weights=True):
        return self.compute_faulted_grid(grid_coord)


    def plot_3d_with_faults(self, t_index=0.0):
        """
        Visualizes the stitched block model along with the extracted 3D fault planes.
        """
        try:
            import pyvista as pv
        except ImportError:
            print("PyVista is required for 3D plotting.")
            return

        print(f"Generating 3D mesh for time index: {t_index}...")
        
        # 1. Setup Grid
        current_section = {4: t_index} 
        full_grid_hyp, final_grid = self.get_section_grid(current_section)
        points = final_grid.numpy()
        nx, ny, nz = self.resolution[0], self.resolution[1], self.resolution[2]
        
        # We increase Z-scale here to act as a Vertical Exaggeration in the plot
        plotter = pv.Plotter(window_size=[1024, 768])
        plotter.set_scale(zscale=2.0) 
        
        # =========================================================
        # 2. EXTRACT AND PLOT FAULT PLANES
        # =========================================================
        fault_colors = ["#ff4500", "#a9a9a9", "#2f4f4f"] # Orange-Red, Grey, Dark Grey
        
        combined_backup = self.save_internal_state()
        try:
            for i, f_name in enumerate(self.fault_names):
                # Load the specific mathematical state for this fault
                self.load_internal_state(self.fault_states[f_name])
                out, _ = super(GempyMultiFaultModel, self).Solution_grid(full_grid_hyp, section_plot=True, recompute_weights=False)
                f_scalar = out["Regular"].numpy()
                f_thresh = self.fault_thresholds[f_name].item()
                
                # Contour the fault scalar field exactly at its mathematical threshold
                f_grid = pv.StructuredGrid()
                f_grid.points = points
                f_grid.dimensions = [nx, ny, nz]
                f_grid["Scalar"] = f_scalar
                
                # Extract the fault plane
                f_surf = f_grid.contour(isosurfaces=[f_thresh])
                c = fault_colors[i % len(fault_colors)]
                plotter.add_mesh(f_surf, color=c, opacity=0.6, label=f"Fault: {f_name}")
        finally:
            # Restore the master state
            self.load_internal_state(combined_backup)

        # =========================================================
        # 3. EXTRACT AND PLOT STRUCTURAL INTERFACES (Rock layers)
        # =========================================================
        # This evaluates the final "Stitched" block model
        _, results = self.Solution_grid(grid_coord=full_grid_hyp, section_plot=True, recompute_weights=False)
        values = torch.round(results['Regular']).numpy()
        
        vol_grid = pv.StructuredGrid()
        vol_grid.points = points
        vol_grid.dimensions = [nx, ny, nz]
        vol_grid["Lithology"] = values
        
        max_layer_val = values.max()
        if max_layer_val >= 2:
            # Contour halfway between Rock IDs (e.g., 1.5, 2.5) to find the layer boundaries
            contour_levels = [i + 0.5 for i in range(1, int(max_layer_val) + 1)]
            interfaces = vol_grid.contour(isosurfaces=contour_levels)
            plotter.add_mesh(interfaces, cmap="viridis", opacity=0.9, label="Rock Interfaces")
            
        # =========================================================
        # 4. PLOT INPUT POINTS
        # =========================================================
        for key, coords in self.sp_coord.items():
            # Remove 4D time coordinate for 3D plotting
            valid_coords = coords[:,[0,1,2]].numpy() if coords.shape[1] > 3 else coords.numpy()
            
            # Color faults black, and rock structures blue
            is_fault = key in self.fault_names
            color = "black" if is_fault else "blue"
            plotter.add_points(valid_coords, color=color, point_size=12, render_points_as_spheres=True)

        plotter.add_axes()
        plotter.add_legend()
        plotter.show_grid()
        plotter.show()

    def plot_3d_with_faults(self, t_min=-0.5, t_max=4.5, t_initial=0.0):
        """
        Visualizes the stitched block model along with the extracted 3D fault planes
        with an interactive time slider.
        """
        try:
            import pyvista as pv
        except ImportError:
            print("PyVista is required for 3D plotting.")
            return

        plotter = pv.Plotter(window_size=[1024, 768])
        plotter.set_scale(zscale=2.0) 
        
        nx, ny, nz = self.resolution[0], self.resolution[1], self.resolution[2]
        fault_colors = ["#ff4500", "#a9a9a9", "#2f4f4f"] # Orange-Red, Grey, Dark Grey

        # =========================================================
        # 1. THE CALLBACK FUNCTION (Runs every time slider moves)
        # =========================================================
        def update_time(value):
            t_index = value
            current_section = {4: t_index} 
            full_grid_hyp, final_grid = self.get_section_grid(current_section)
            points = final_grid.numpy()

            # --- EXTRACT AND PLOT FAULT PLANES ---
            combined_backup = self.save_internal_state()
            try:
                for i, f_name in enumerate(self.fault_names):
                    self.load_internal_state(self.fault_states[f_name])
                    out, _ = super(GempyMultiFaultModel, self).Solution_grid(full_grid_hyp, section_plot=True, recompute_weights=False)
                    f_scalar = out["Regular"].numpy()
                    f_thresh = self.fault_thresholds[f_name].item()
                    
                    f_grid = pv.StructuredGrid()
                    f_grid.points = points
                    f_grid.dimensions = [nx, ny, nz]
                    f_grid["Scalar"] = f_scalar
                    
                    try:
                        # Contour the fault
                        f_surf = f_grid.contour(isosurfaces=[f_thresh])
                        c = fault_colors[i % len(fault_colors)]
                        # The 'name' argument is crucial: it replaces the old mesh instead of drawing over it
                        plotter.add_mesh(f_surf, color=c, opacity=0.6, name=f"fault_mesh_{f_name}", label=f"Fault: {f_name}" if value==t_initial else None)
                    except ValueError:
                        # If the fault completely disappears at this time step, contour() throws an error.
                        # We catch it and remove the mesh from the scene.
                        plotter.remove_actor(f"fault_mesh_{f_name}")

            finally:
                self.load_internal_state(combined_backup)

            # --- EXTRACT AND PLOT STRUCTURAL INTERFACES ---
            _, results = self.Solution_grid(grid_coord=full_grid_hyp, section_plot=True, recompute_weights=False)
            values = torch.round(results['Regular']).numpy()
            
            vol_grid = pv.StructuredGrid()
            vol_grid.points = points
            vol_grid.dimensions = [nx, ny, nz]
            vol_grid["Lithology"] = values
            
            max_layer_val = values.max()
            if max_layer_val >= 2:
                contour_levels = [i + 0.5 for i in range(1, int(max_layer_val) + 1)]
                try:
                    interfaces = vol_grid.contour(isosurfaces=contour_levels)
                    plotter.add_mesh(interfaces, cmap="viridis", opacity=0.9, name="rock_interfaces_mesh", label="Rock Interfaces" if value==t_initial else None)
                except ValueError:
                    plotter.remove_actor("rock_interfaces_mesh")

        # =========================================================
        # 2. PLOT STATIC INPUT POINTS
        # =========================================================
        # We plot these outside the callback because they do not change shape when the slider moves
        for key, coords in self.sp_coord.items():
            valid_coords = coords[:,[0,1,2]].numpy() if coords.shape[1] > 3 else coords.numpy()
            is_fault = key in self.fault_names
            color = "black" if is_fault else "blue"
            plotter.add_points(valid_coords, color=color, point_size=12, render_points_as_spheres=True, name=f"points_{key}")

        # =========================================================
        # 3. INITIALIZE PLOT & ADD SLIDER
        # =========================================================
        # Run once to draw the initial state
        update_time(t_initial)
        
        # Add the interactive slider
        plotter.add_slider_widget(
            callback=update_time, 
            rng=[t_min, t_max], 
            value=t_initial, 
            title="Evolution Time (t)", 
            pointa=(0.025, 0.1), # Position bottom left
            pointb=(0.31, 0.1),
            style='modern'
        )

        plotter.add_axes()
        plotter.add_legend()
        plotter.show_grid()
        plotter.show()

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    

    ## DYNAMIC ISOTROPIC SCALING 
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    #TRIAL -1: No fault data at time T = 5.0, only flattening layers
    df_sp = pd.read_csv(os.path.join(BASE_DIR, "trial1_surfaces.csv"))
    df_op = pd.read_csv(os.path.join(BASE_DIR, "trial1_orientations.csv"))

    # Find the absolute real-world boundaries of the data
    min_x, max_x = df_sp['X'].min(), df_sp['X'].max()
    min_y, max_y = df_sp['Y'].min(), df_sp['Y'].max()
    min_z, max_z = df_sp['Z'].min(), df_sp['Z'].max()
    
    # Calculate the maximum span to ensure angles are not distorted
    max_dist = max(max_x - min_x, max_y - min_y, max_z - min_z)

    #Create a scaling function that shifts data to 0.0 and scales by max_dist
    def scale_coords(x, y, z):
        x_s = (x - min_x) / max_dist
        y_s = (y - min_y) / max_dist
        z_s = (z - min_z) / max_dist 
        return x_s, y_s, z_s

    # Apply scaling to the dataframes
    df_sp['X_s'], df_sp['Y_s'], df_sp['Z_s'] = scale_coords(df_sp['X'], df_sp['Y'], df_sp['Z'])
    df_op['X_s'], df_op['Y_s'], df_op['Z_s'] = scale_coords(df_op['X'], df_op['Y'], df_op['Z'])
    
    
    # Dynamically set the Model Extent with a 5% buffer so data doesn't touch the walls
    buffer = 0.05
    extent = [
        0.0 - buffer-0.1,
          ((max_x - min_x) / max_dist) + buffer+0.1,
        0.0 - buffer-0.1, ((max_y - min_y) / max_dist) + buffer+0.1,

        # 0.0 - buffer, ((max_y - min_y) / max_dist),

        0.0 - buffer-0.1, ((max_z - min_z) / max_dist) + buffer+0.1,
        # 0.0 - buffer,
        #   ((max_z - min_z) / max_dist)*1.5 +0.15,

        0.0, 5.0 
    ]
    
    # print(f"Calculated  Extent: {extent}")

    resolution = [60, 50, 60, 10]

    # Initialize Model
    model = GempyMultiFaultModel("Gullfaks 4D Model", extent, resolution)
    
    # =========================================
    #  PREPARING DATA
    # =========================================
    def get_4d_sp(df, formation):
        subset = df[df['formation'] == formation]
        if subset.empty: return torch.empty(0)
        
        # USING MANUALLY SCALED COLUMNS
        # Pulling directly from our dynamically generated 'T' column
        coords = subset[['X_s', 'Y_s', 'Z_s', 'T']].values
        return torch.tensor(coords, dtype=torch.float32)

    def get_4d_op(df, formations):
        if isinstance(formations, str): formations = [formations]
        subset = df[df['formation'].isin(formations)]
        if subset.empty: return {'Positions': torch.empty(0), 'Values': torch.empty(0)}
        
        pos = subset[['X_s', 'Y_s', 'Z_s', 'T']].values
        vals = subset[['G_x', 'G_y', 'G_z', 'G_t']].values
        
        return {
            'Positions': torch.tensor(pos, dtype=torch.float32),
            'Values': torch.tensor(vals, dtype=torch.float32)
        }

    # =====================================================================
    # FAULT & STRUCTURE INPUT SETUP
    # =====================================================================
    
    fault4orientation_data = get_4d_op(df_op, 'fault4')
    # Faults
    fault4_input = {
        'sp_coord': {'fault4': get_4d_sp(df_sp, 'fault4')}, 
        'op_coord': fault4orientation_data, 
        'transformation_matrix': torch.diag(torch.tensor([1,1,1,0.05]))
    }

    

    fault3orientation_data = get_4d_op(df_op, 'fault3')

    fault3_input = {
        'sp_coord': {'fault3': get_4d_sp(df_sp, 'fault3')}, 
        'op_coord': fault3orientation_data, 
        'transformation_matrix': torch.diag(torch.tensor([1,1,1,0.05]))
    }

    faults_dict = {'fault4': fault4_input, 'fault3': fault3_input}

    # Structure
    struct_formations = ['etive', 'ness','tarbert']
    struct_interface_data = {fmt: get_4d_sp(df_sp, fmt) for fmt in struct_formations if len(get_4d_sp(df_sp, fmt)) > 0}
    struct_orientation_data = get_4d_op(df_op, struct_formations)
    
    struct_transformation_matrix = torch.diag(torch.tensor([1,1,1,0.05]))

    struct_input = {
        'sp_coord': struct_interface_data, 
        'op_coord': struct_orientation_data, 
        'transformation_matrix': struct_transformation_matrix
    }

    print("Computing models...")
    model.compute_models(faults_data=faults_dict, structure_data=struct_input)
    
    # --- PLOTTING ---
    #########################################################################
    ###### Uncomment the below code lines for matplotlib visualization ######
    #########################################################################

    #### FOR 2D matplotlib #####
    import time
    for t in [0,1,2,3,4,5,6,7,8,9,10]:
        print(f"TIme:{t}")
        model.plot_data_section(section={2:0.2, 4:t}, plot_scalar_field = True, plot_input_data=False)
        time.sleep(1)


    ### FOR 3D matplotlib #####
    import time
    for t in [0,1,2,3,4,5,6,7,8,9,10]:
        model.plot_data_section(section={4:t}, plot_scalar_field = True, plot_input_data=False)
        time.sleep(1)


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


    # New Multi-Fault 3D Plot - uncomment below

    # model.plot_3d_with_faults(t_min=-0.0, t_max=10.0, t_initial=0.0)