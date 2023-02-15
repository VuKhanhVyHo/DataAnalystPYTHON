ACC_DUE_TO_GRAV = 9.81

MAIN_PROMPT = """Please enter a command: """

HELP = """
The available commands are:
    'h' - Displays the help text.
    's' - Repleace existing runway and input parameters. 
        Adds to existing plane parameters.
    's i' - to specify input parameters only. Replaces existing input 
        parameters.
    'p <First runway> <Last runway>' - "p 1 2" prints out the liftoff 
        distances on Runways 1 and 2; "p 1 1" prints out the liftoff distances 
        on Runway 1 only.
    'r <r or p>' - 
        'r r' read runway parameters from a file and replace existing runway 
            parameters.
        'r p' read plane parameters from a file and add to existing plane 
            parameters.
    'q' - quit.
"""

RUNWAY_PROMPTS = (
    "length of the runway (km)",
    "slope of the runway (rad)",
)

PLANE_PROMPTS = (
    "mass of the plane (in kg)",
    "engine thrust (in N)",
    "reference area (in m^2)",
    "lift-off velocity (in m/s)",
    "drag coefficient",
)

INPUT_PROMPTS = (
    "air density (in kg/m^3)",
    "initial velocity (v_0) at start of runway (in m/s)",
    "position (x_0) at the start of the runway (in m)",
    "time increment (in secs)"
)

ALL_PROMPTS = RUNWAY_PROMPTS + PLANE_PROMPTS + INPUT_PROMPTS

INVALID = "Please enter a valid command."

SEPARATOR = "#"

def load_data(directory: str, file_name: str) -> tuple[tuple[float, ...], ...]:
    """ Reads in a data file, converts the information into floating point
        numbers, then returns the numbers in tuples.
        Parameters:
            directory (str): The name of the directory where the data files are
                stored
            file_name (str): The name of the specific file to be loaded in
        Returns:
            (tuple[tuple[float, ...], ...]): The data from the file with
                each row of data as a tuple of floats.
    """
    output = ()
    with open(directory + '/' + file_name, 'r') as file:
        for line in file.readlines():
            output += (tuple([float(x) for x in line.strip().split(', ')]),)

    return output
