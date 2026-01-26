# GALFORM galaxies.hdf5 Parameters Documentation

This document explains all parameters in the GALFORM `galaxies.hdf5` output files based on analysis of the data and GALFORM source code.

## Key Finding: Halo Hierarchy and Galaxy Types

**IMPORTANT**: Every object in `galaxies.hdf5` represents a **galaxy**, not necessarily a distinct dark matter halo. The total count (149,285 galaxies in this snapshot) includes:
- **Central galaxies** (`is_central=1`): 96,316 galaxies - one per (sub)halo
- **Satellite galaxies** (`is_central=0`): 52,969 galaxies - additional galaxies within halos

## Halo Identification Fields

### `is_central` (integer: 0 or 1)
**Definition**: Flag indicating whether a galaxy is central or satellite
- **1**: Central galaxy - the primary galaxy at the center of its (sub)halo
- **0**: Satellite galaxy - an additional galaxy within a halo (result of mergers)

**Source**: `output.write_galaxies.F90`
```fortran
if (This_Node%jlevel.eq.levout) then
    is_central=1  ! Central galaxy
else
    is_central=0  ! Satellite galaxy
end if
```

**Usage**: For correlation functions, use `is_central==1` to select one representative per halo.

### `hierarchy_level` (integer: 0-5)
**Definition**: Position in the halo substructure hierarchy
- **0**: Isolated halo (top of hierarchy tree) - 96,316 galaxies
- **1**: Sub-halo (substructure within a larger halo)
- **2**: Sub-sub-halo
- **3+**: Deeper levels of substructure

**Comment from source**: `"0 = isolated halo, 1 = sub-halo, 2 = sub-sub-halo etc"`

**Key insight**: ALL central galaxies have `hierarchy_level=0`, meaning GALFORM treats each (sub)halo's central galaxy as if it were in an isolated halo at its own level. This does NOT distinguish main halos from subhalos!

### `ihhalo` (integer: halo index)
**Definition**: Index of the **host halo** at this output time
- For centrals: Points to the parent/host halo in the hierarchy
- For satellites: Points to the halo containing this satellite

**Source**: `output.write_galaxies.F90`
```fortran
select case (is_central)
case (0):  ! Satellite
    Parent_Node => Tree_Get_Current_Parent(This_Node,levout)
case (1):  ! Central
    Parent_Node => This_Node
end select
ihhalo=Parent_Node%nodeindex
```

**Range**: 65 to 70,549 (7,268 unique values in this snapshot)
**Interpretation**: This is a **halo ID** reference, NOT a 0/1 flag!

### `ihalof` (integer: formation halo index)
**Definition**: Index of the halo where this galaxy's halo **first formed**
**Source**: `ihalof=This_Node%formation%nodeindex`

**Comment from source**: `"Location in the halo merger tree where the halo in which this galaxy resides formed"`

**Range**: 65 to 1,179,302 (36,700 unique values)

**Key insight**: When `ihalof == ihhalo`, the halo is at its formation location - these are **main/field halos** (595 galaxies). When `ihalof != ihhalo`, the halo has been accreted into a larger structure.

### `index` (integer)
**Definition**: Index identifier for this galaxy/node in the merger tree
**Range**: 65 to 1,178,260 (29,692 unique values)

## Mass Fields

### `mhalo` (float: Msun/h)
**Definition**: **Subhalo mass** - mass of the (sub)halo containing this galaxy
**Range**: 1.59×10⁹ to 2.42×10¹⁴ Msun/h
**Usage**: Use for galaxy-based mass cuts

### `mhhalo` (float: Msun/h)
**Definition**: **Host halo mass** - mass of the ultimate parent halo
- For centrals in main halos: `mhhalo ≈ mhalo`
- For centrals in subhalos: `mhhalo > mhalo` (host is larger)
- For satellites: `mhhalo` is the mass of their host halo

**Range**: 2.12×10⁹ to 4.61×10¹⁴ Msun/h
**Usage**: Use for halo-based mass cuts in correlation functions

### `mchalo` (float: Msun/h)
**Definition**: Alternative halo mass measure (possibly virial mass)
**Range**: 2.02×10⁹ to 4.61×10¹⁴ Msun/h

### `mstars_disk` (float: Msun/h)
**Definition**: Stellar mass in disk component
**Range**: 0 to 4.84×10¹⁰ Msun/h

### `mstars_bulge` (float: Msun/h)
**Definition**: Stellar mass in bulge component
**Range**: 0 to 1.81×10¹¹ Msun/h

### `mstars_allburst` (float: Msun/h)
**Definition**: Total stellar mass from all burst episodes
**Range**: 0 to 1.11×10¹¹ Msun/h

## Position and Velocity Fields

### `xgal`, `ygal`, `zgal` (float: Mpc/h)
**Definition**: 3D comoving position of the galaxy
**Range**: 0 to 542.16 Mpc/h (box size ~542 Mpc/h)
**Note**: Same position arrays used for both galaxies and their host halos

### `vxgal`, `vygal`, `vzgal` (float: km/s)
**Definition**: 3D peculiar velocity of the galaxy
**Range**: ~-3000 to +3000 km/s

### `vhalo`, `vhhalo`, `vchalo` (float: km/s)
**Definition**: Circular velocities
- `vhalo`: Subhalo circular velocity
- `vhhalo`: Host halo circular velocity  
- `vchalo`: Another halo velocity measure

## Size Fields

### `rdisk` (float: Mpc/h)
**Definition**: Disk scale length
**Range**: 0 to 0.0312 Mpc/h

### `rbulge` (float: Mpc/h)
**Definition**: Bulge effective radius
**Range**: 0 to 0.0150 Mpc/h

### `rcomb` (float: Mpc/h)
**Definition**: Combined radius measure
**Range**: 4.04×10⁻⁵ to 0.0269 Mpc/h

### `halo_r_virial` (float: Mpc/h)
**Definition**: Virial radius of the halo
**Range**: 0.00309 to 1.48 Mpc/h

## Star Formation Fields

### `mstardot` (float: Msun/h/Gyr)
**Definition**: Current star formation rate
**Range**: 0 to 2.81×10¹⁰ Msun/h/Gyr

### `mstardot_average` (float: Msun/h/Gyr)
**Definition**: Time-averaged star formation rate
**Range**: 0 to 2.82×10¹⁰ Msun/h/Gyr

### `mstardot_burst` (float: Msun/h/Gyr)
**Definition**: Star formation rate in burst mode
**Range**: 0 to 7.95×10¹⁰ Msun/h/Gyr
**Unique values**: Only 6 (discrete burst episodes)

## Gas Mass Fields

### `mcold` (float: Msun/h)
**Definition**: Total cold gas mass
**Range**: 0 to 5.82×10¹⁰ Msun/h

### `mcold_atom` (float: Msun/h)
**Definition**: Atomic cold gas mass
**Range**: 0 to 2.78×10¹⁰ Msun/h

### `mcold_mol` (float: Msun/h)
**Definition**: Molecular cold gas mass
**Range**: 0 to 4.11×10¹⁰ Msun/h

### `mhot` (float: Msun/h)
**Definition**: Hot gas mass in halo
**Range**: 0 to 6.85×10¹³ Msun/h

## Angular Momentum Fields

### `angmom_disk` (float: Msun/h Mpc/h km/s)
**Definition**: Angular momentum of stellar disk
**Range**: 0 to 2.79×10¹¹

### `angmom_dm` (float: Msun/h Mpc/h km/s)
**Definition**: Angular momentum of dark matter halo
**Range**: 1.42×10⁷ to 4.33×10¹⁶

### `spin` (float: dimensionless)
**Definition**: Halo spin parameter
**Range**: 0.00353 to 0.200

### `subhalo_spin_x`, `subhalo_spin_y`, `subhalo_spin_z` (float)
**Definition**: Components of subhalo spin vector
**Range**: ~-30 to +30

## Time-related Fields

### `redshift` (float: scalar)
**Definition**: Redshift of this output snapshot
**Value**: 0.4959 (constant for all galaxies in this snapshot)

### `achalo` (float: expansion factor)
**Definition**: Expansion factor when halo was last central
**Range**: 0.0778 to 0.668

### `aform` (float: expansion factor)
**Definition**: Expansion factor when halo formed
**Range**: 0.0643 to 0.668

### `tburst` (float: Gyr)
**Definition**: Time since last burst
**Range**: 0.00630 to 100,000 Gyr

### `tsink` (float: Gyr)
**Definition**: Dynamical timescale or sink time
**Range**: -1 to 82,118 Gyr

## Merger Tree Fields

### `SubhaloID` (int64)
**Definition**: Unique identifier for subhalo across snapshots
**Range**: 5.10×10¹³ to 4.02×10¹⁶

### `SubhaloIndex` (int32)
**Definition**: Index of subhalo in merger tree
**Range**: 605 to 2.22×10⁸

### `SubhaloSnapNum` (int32)
**Definition**: Snapshot number where subhalo first appears
**Range**: 51 to 205

### `ParticleID` (int64)
**Definition**: ID of representative N-body particle
**Range**: 76,319 to 1.28×10¹¹

## Metallicity Fields

### `cold_metal` (float: Msun/h)
**Definition**: Metal mass in cold gas
**Range**: 0 to 1.06×10⁹ Msun/h

### `hot_metal` (float: Msun/h)
**Definition**: Metal mass in hot gas
**Range**: 0 to 8.32×10¹⁰ Msun/h

### `star_metal_disk` (float: Msun/h)
**Definition**: Metal mass in disk stars
**Range**: 0 to 1.29×10⁹ Msun/h

### `star_metal_bulge` (float: Msun/h)
**Definition**: Metal mass in bulge stars
**Range**: 0 to 3.78×10⁹ Msun/h

## Black Hole Fields

### `M_SMBH` (float: Msun/h)
**Definition**: Supermassive black hole mass
**Range**: 0 to 2.89×10⁹ Msun/h

### `SMBH_Mdot_hh` (float: Msun/h/yr)
**Definition**: BH accretion rate from hot halo mode
**Range**: 0 to 3.90×10⁸ Msun/h/yr

### `SMBH_Mdot_stb` (float: Msun/h/yr)
**Definition**: BH accretion rate from starburst mode
**Range**: 0 to 4.12×10¹¹ Msun/h/yr

### `SMBH_Spin` (float: dimensionless)
**Definition**: Black hole spin parameter
**Value**: 0 (not evolved in this run)

## Burst/Merger Fields

### `burst_mode` (integer: 0, 1, or 2)
**Definition**: Current burst activity mode
**Values**:
- 0: No burst
- 1: Minor merger burst
- 2: Major merger burst

### `mburst` (float: Msun/h)
**Definition**: Mass involved in current burst
**Range**: 0 to 8.13×10¹⁰ Msun/h

### `GalaxyMergerOn` (float: 0 or 1)
**Definition**: Flag for whether galaxy is currently merging
**Values**: 0 (no merger) or 1 (merger in progress)

## Velocity Dispersion Fields

### `vbulge` (float: km/s)
**Definition**: Bulge velocity dispersion
**Range**: 0 to 1,950 km/s

### `vdisk` (float: km/s)
**Definition**: Disk circular velocity
**Range**: 0 to 1,812 km/s

## Structural Parameters

### `strc` (float: dimensionless)
**Definition**: NFW concentration parameter
**Range**: 0.0963 to 0.254

## Type Classification

### `type` (integer: 0, 1, or 2)
**Definition**: Galaxy morphological type
**Values**: 0, 1, 2 (specific meanings depend on GALFORM setup)

