"""Module for reading and writing Specfem parameter files

Originally taken from GF3D.

"""
from copy import deepcopy
from collections import OrderedDict


def par2par_file(parameter: float | str | int | bool) -> str:
    """Converts a value to a specfem dictionary parameter

    Parameters
    ----------
    parameter : float | str | int | bool
        parameter value

    Returns
    -------
    str
        outputstring for writing the parameter file

    Raises
    ------
    ValueError
        raised if parameter conversion not implemented
    """

    if isinstance(parameter, float):

        # Check whether g formatting removes decimal points
        out = f'{parameter:g}'
        if '.' in out:
            out += 'd0'
        else:
            out += '.d0'

        return out

    elif isinstance(parameter, str):
        return parameter

    elif isinstance(parameter, bool):
        if parameter:
            return '.true.'
        else:
            return '.false.'

    elif isinstance(parameter, int):
        return f'{parameter:d}'

    else:
        raise ValueError(
            f'Parameter conversion not implemented for\n'
            f'P: {parameter} of type {type(parameter)}')


def checkInt(string):
    """Checks whether a string is an integer

    Parameters
    ----------
    string : str
        input string

    Returns
    -------
    bool
        True if string is an integer, False otherwise
    """
    if string[0] in ('-', '+'):
        return string[1:].isdigit()
    else:
        return string.isdigit()


def par_file2par(value: str, verbose: bool = False) -> float | str | int | bool:
    """Converts a value from a parameter file to a python value

    Parameters
    ----------
    value : str
        input value as string from parameter file
    verbose : bool, optional
        Increase verbosity, by default False

    Returns
    -------
    float | str | int | bool
        output value
    """

    if value == ".true.":
        rvalue = True

    elif value == ".false.":
        rvalue = False

    elif "d0" in value:
        rvalue = float(value.replace("d0", "0"))

    elif checkInt(value):
        rvalue = int(value)

    else:
        rvalue = value

    if verbose:
        print(f'converting {value} to {rvalue}')

    return rvalue

def print_model_vector(model_vector: list):
    """Prints a model vector in a readable way

    Parameters
    ----------
    model_vector : list
        model vector
        
        
    Note
    ---
    
    Documentation of model vector from the ``Par_file``
    
    .. code-block::
    
        #   acoustic:    model_number 1 rho Vp 0  0 0 QKappa Qmu 0 0 0 0 0 0
        #   elastic:     model_number 1 rho Vp Vs 0 0 QKappa Qmu 0 0 0 0 0 0
        #   anistoropic: model_number 2 rho c11 c13 c15 c33 c35 c55 c12 c23 c25 0 0 0
        #   poroelastic: model_number 3 rhos rhof phi c kxx kxz kzz Ks Kf Kfr etaf mufr Qmu
        #   tomo:        model_number -1 0 9999 9999 A 0 0 9999 9999 0 0 0 0 0

    """
    
    if model_vector[0] == 1:
        print(f'Acoustic model with rho={model_vector[1]}, Vp={model_vector[2]}, QKappa={model_vector[7]}, Qmu={model_vector[8]}')
    elif model_vector[0] == 2:
        print(f'Elastic model with rho={model_vector[1]}, Vp={model_vector[2]}, Vs={model_vector[3]}, QKappa={model_vector[7]}, Qmu={model_vector[8]}')
    elif model_vector[0] == 3:
        print(f'Anisotropic model with rho={model_vector[1]}, c11={model_vector[2]}, c13={model_vector[3]}, c15={model_vector[4]}, c33={model_vector[5]}, c35={model_vector[6]}, c55={model_vector[7]}, c12={model_vector[8]}, c23={model_vector[9]}, c25={model_vector[10]}')
    elif model_vector[0] == 4:
        print(f'Poroelastic model with rhos={model_vector[1]}, rhof={model_vector
        [2]}, phi={model_vector[3]}, c={model_vector[4]}, kxx={model_vector[5]}, kxz={model_vector[6]}, kzz={model_vector[7]}, Ks={model_vector[8]}, Kf={model_vector[9]}, Kfr={model_vector[10]}, etaf={model_vector[11]}, mufr={model_vector[12]}, Qmu={model_vector[13]}')
    elif model_vector[0] == -1:
        print(f'Tomo model.')
    else:
        raise ValueError('Model not recognized.')
    
    
def print_region(region: list):
    
    print(f'Region with latmin=')


def get_par_file(parfile: str, savecomments: bool = False, verbose: bool = True) -> OrderedDict:
    """Reads a Specfem parameter file and returns an ordered dictionary.
    it can preserve comments and empty lines.

    Parameters
    ----------
    parfile : str
        Path to the Par_file
    savecomments : bool, optional
        _description_, by default False
    verbose : bool, optional
        _description_, by default True

    Returns
    -------
    OrderedDict
        dictionary with parameters

    Raises
    ------
    ValueError
        raised if conversion of line not implemented
    """

    pardict = OrderedDict()
    cmtcounter = 0
    noncounter = 0
    cmtblock = []
    nonblock = []
    models = []
    model_counter = 0
    regions = []
    region_counter = 0

    with open(parfile, 'r') as f:
        for line in f.readlines():

            if verbose:
                print(line.strip())

            # Check for comment by removing leading (all) spaces
            if '#' == line.replace(' ', '')[0]:
                if savecomments:

                    if len(nonblock) > 0:
                        pardict[f'space-{noncounter}'] = nonblock
                        nonblock = []
                        noncounter += 1

                    cmtblock.append(line)

            # Check for empty line by stripping all spaces except '\n'
            elif '\n' == line.replace(' ', ''):

                if savecomments:
                    if len(cmtblock) > 0:
                        pardict[f'comment-{cmtcounter}'] = cmtblock
                        cmtblock = []
                        cmtcounter += 1

                    nonblock.append(line)

            elif '=' in line:

                if savecomments:

                    if len(cmtblock) > 0:
                        pardict[f'comment-{cmtcounter}'] = cmtblock
                        cmtblock = []
                        cmtcounter += 1

                    if len(nonblock) > 0:
                        pardict[f'space-{noncounter}'] = nonblock
                        nonblock = []
                        noncounter += 1

                # Get key and value
                key, val = line.split('=')[:2]
                key = key.replace(' ', '')

                # Save guard if someone puts a comment behind a value
                if '#' in val:
                    val, cmt = val.split('#')
                else:
                    cmt = None

                val = val.strip()

                # Add key and value to dictionary
                pardict[key] = par_file2par(val, verbose=verbose)

                # save comment behind value
                if savecomments:
                    if cmt is not None:
                        pardict[f'{key}-comment'] = cmt.strip()
                        
            elif checkInt(line[0]):
                
                # split line into list of strings
                line = line.strip().split()
                
                # Ignore comment at the of the model/region lines
                if '#' in line:
                    line = line[:line.index('#')]
                
                if len(line) == 15:
                    
                
                    model_counter += 1
                    
                    if savecomments:
                        if len(cmtblock) > 0:
                            pardict[f'comment-{cmtcounter}'] = cmtblock
                            cmtblock = []
                            cmtcounter += 1

                        if len(nonblock) > 0:
                            pardict[f'space-{noncounter}'] = nonblock
                            nonblock = []
                            noncounter += 1

                    if verbose:
                        print(f'Converting model {model_counter} to list')

                    
                    # Convert each model parameter to python value
                    model = [par_file2par(x) for x in line]
                
                    # Append model to list of models
                    models.append(model)

                    # Print model
                    if verbose:
                        print_model_vector(model)
                        
                elif len(line) == 5:
                    
                    region_counter += 1
                    
                    if savecomments:
                        if len(cmtblock) > 0:
                            pardict[f'comment-{cmtcounter}'] = cmtblock
                            cmtblock = []
                            cmtcounter += 1

                        if len(nonblock) > 0:
                            pardict[f'space-{noncounter}'] = nonblock
                            nonblock = []
                            noncounter += 1

                    if verbose:
                        print(f'Converting region {region_counter} to list')

                    # Convert each model parameter to python value
                    region = [par_file2par(x) for x in line]
                
                    # Append model to list of models
                    regions.append(region)

                    # Print model
                    if verbose:
                        print_region(region)
                        
                else:
                    raise ValueError(f'Could not convert model or region {line} to list.')

            else:
                raise ValueError(f'Conversion of\n{line}\n not implemented.')

    # Add models to dictionary or throw error if no models found
    if len(models) > 0:
        pardict['models'] = models
        
    elif len(models) != pardict.nbmodels:
        raise ValueError('No models found in parfile.')
    
    # Add regions to dictionary or throw error if no regions found
    if len(regions) > 0:
        pardict['regions'] = regions
        
    elif len(regions) != pardict.nbregions:
        raise ValueError('No regions found in parfile.')
    
    return pardict


def write_par_file(pardict: OrderedDict, par_file: str | None = None, write_comments: bool = True):
    
    # Deep copy the dictionary
    pardict = deepcopy(pardict)

    # If output file is provided open a file to write
    if par_file is not None:
        f = open(par_file, 'w')
    else:
        f = None

    for key, value in pardict.items():

        if 'comment-' in key or 'space-' in key:

            if write_comments:
                if f is not None:
                    f.writelines(value)
                    # f.write('\n')
                else:
                    for line in value:
                        print(line.strip())
                        
        elif key=='nbmodels':
            
            # Check whether nbmodels and models have the same length
            if len(pardict["models"]) != value:
                raise ValueError('Number of models does not match nbmodels.')
            
            # Write nbmodels
            outstr = f"{key:31s} = {par2par_file(value):<s}\n"
            
            # Write models
            for model in pardict["models"]:
                outstr += ' '.join([par2par_file(x) for x in model]) + '\n'
                
            f.write(outstr)

        elif key=='nbregions':
            
            # Check whether nbregions and regions have the same length
            if len(pardict["regions"]) != value:
                raise
            
            # Write nbregions
            outstr = f"{key:31s} = {par2par_file(value):<s}\n"
            
            # Write regions
            for region in pardict["regions"]:
                outstr += ' '.join([par2par_file(x) for x in region]) + '\n'
            
            f.write(outstr)
            
        elif key=='models' or key=='regions':
            continue
            
        else:

            # Skip value comments
            if f'-comment' in key:
                continue

            # Fix the parfile print depending on whether value has comment
            try:
                if f'{key}-comment' in pardict and write_comments:
                    outstr = f'{key:31s} = {par2par_file(value):<s}   # {pardict[f"{key}-comment"]:<s}\n'
                else:
                    outstr = f"{key:31s} = {par2par_file(value):<s}\n"
            except ValueError as e:
                raise ValueError(
                    f'Key: {key} has value {value}. Par2parfile is not implemented for that type.')

            # Print string or write to file
            if f is not None:
                f.write(outstr)
            else:
                print(outstr.strip())

    # Close file if defined
    if f is not None:
        f.close()

