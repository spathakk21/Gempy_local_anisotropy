 import time
    for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 3.5, 4,4.5]:
        gp.plot_data_section(section={4:t}, plot_scalar_field = True, plot_input_data=True)
        time.sleep(1)