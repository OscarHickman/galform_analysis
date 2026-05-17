import json

with open("examples/redshift_space_distortions/scope_rsd_kaiser.ipynb") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        
        # 1. Add MAX_GAL_PER_SUBVOL constant
        if "MHALO_MIN  = 1e10" in src and "MAX_GAL_PER_SUBVOL" not in src:
            src = src.replace("MHALO_MIN  = 1e10        # M_sun/h", "MHALO_MIN  = 1e10        # M_sun/h\nMAX_GAL_PER_SUBVOL = 3000\n")
            
        # 2. Add downsampling logic in the first loop
        if 'arr, meta = read_galaxy_arrays(' in src:
            if 'idx = np.random.choice' not in src:
                target = "vz = np.asarray(arr[\"vzgal\"], dtype=np.float64)\n"
                replacement = target + """
    n = len(x)
    if n > MAX_GAL_PER_SUBVOL:
        idx = np.random.choice(n, size=MAX_GAL_PER_SUBVOL, replace=False)
        x, y, z, vz = x[idx], y[idx], z[idx], vz[idx]
"""
                src = src.replace(target, replacement)

        # 3. Add downsampling logic in the second loop (which uses z_)
        if 'arr, _ = read_galaxy_arrays(' in src:
            if 'idx = np.random.choice' not in src:
                target2 = "vz = np.asarray(arr[\"vzgal\"], dtype=np.float64)\n"
                replacement2 = target2 + """
        n = len(x)
        if n > MAX_GAL_PER_SUBVOL:
            idx = np.random.choice(n, size=MAX_GAL_PER_SUBVOL, replace=False)
            x, y, z_, vz = x[idx], y[idx], z_[idx], vz[idx]
"""
                src = src.replace(target2, replacement2)
                
        # Re-apply modified source to the cell (handling newlines safely)
        lines = src.split('\n')
        # Reconstruct list of strings with newlines at the end of each except the last
        cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open("examples/redshift_space_distortions/scope_rsd_kaiser.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
