! Command-line wrapper around GALFORM's diffuse-dust routines.
!
! galform_analysis compiles this file against the user's own GALFORM build
! and runs it. It contains no dust model: all the physics is in GALFORM.
!
! Usage: galform_dust_driver <dust table> <input file> <output file>
!
! Input (stream, native byte order): n, nlam, ave (int32); h0 (real32);
! lambda(nlam); then n values each of mdisk, zdisk, rdisk, rbulge, mburst,
! zburst, rburst, inc (real32) and burst (int32, 0 or 1).
! Output: atten(2, nlam, n) and tauv0(n) (real32).
program galform_dust_driver
  use Cosmology, only: h0
  use Dust_Module, only: Dust_Read
  use Dust_Attenuation, only: Calc_Atten_Diffuse
  implicit none
  character(len=4096) :: table(1), infile, outfile
  integer :: n, nlam, ave, i, unit_in, unit_out
  real :: h0_in, tauv0_burst
  real, allocatable :: lambda(:), mdisk(:), zdisk(:), rdisk(:), rbulge(:)
  real, allocatable :: mburst(:), zburst(:), rburst(:), inc(:)
  real, allocatable :: atten(:, :, :), tauv0(:)
  integer, allocatable :: burst(:)

  call get_command_argument(1, table(1))
  call get_command_argument(2, infile)
  call get_command_argument(3, outfile)

  open (newunit=unit_in, file=trim(infile), access='stream', form='unformatted', status='old')
  read (unit_in) n, nlam, ave, h0_in
  allocate (lambda(nlam), mdisk(n), zdisk(n), rdisk(n), rbulge(n), mburst(n), zburst(n), &
            rburst(n), inc(n), burst(n), atten(2, nlam, n), tauv0(n))
  read (unit_in) lambda, mdisk, zdisk, rdisk, rbulge, mburst, zburst, rburst, inc, burst
  close (unit_in)

  h0 = h0_in
  call Dust_Read(1, table)
  do i = 1, n
    call Calc_Atten_Diffuse(1, mdisk(i), zdisk(i), rdisk(i), rbulge(i), burst(i) == 1, &
                            mburst(i), zburst(i), rburst(i), inc(i), ave == 1, lambda, nlam, &
                            atten(:, :, i), tauv0(i), tauv0_burst)
  end do

  open (newunit=unit_out, file=trim(outfile), access='stream', form='unformatted', status='replace')
  write (unit_out) atten, tauv0
  close (unit_out)
end program galform_dust_driver
