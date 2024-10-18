program coordinate_transform
  use grid_module
  use transform_module
  use cudafor
  implicit none

  integer, parameter :: nx = 10000, ny = 10000
  integer :: nxny
  real(8), allocatable :: x_host(:), y_host(:)
  real(8), allocatable, device :: x_in_dev(:), y_in_dev(:)
  real(8), allocatable, device :: x_out_dev(:), y_out_dev(:)
  real(8), allocatable :: x_transformed(:), y_transformed(:)
  real(8) :: x_obs, y_obs
  integer :: istat
  !type(cudaStream) :: stream1, stream2
  integer(kind=cuda_stream_kind) :: stream1, stream2

  integer :: threadsPerBlock, blocksPerGrid

  nxny = nx * ny
  allocate(x_host(nxny), y_host(nxny))
  allocate(x_transformed(nxny), y_transformed(nxny))
  allocate(x_in_dev(nxny), y_in_dev(nxny))
  allocate(x_out_dev(nxny), y_out_dev(nxny))

  ! generate the grid of points to operate over
  call generate_grid (x_host, y_host, nx, ny, 1.0d0, 1.0d0)

  ! Placing some observer
  x_obs = nx * 2.5d0
  y_obs = ny * 2.5d0

  ! comment this
  istat = cudaStreamCreate(stream1)
  istat = cudaStreamCreate(stream2)

  istat = cudaMemcpyAsync(x_in_dev, x_host, nxny * sizeof(x_host(1)), cudaMemcpyHostToDevice, stream1)
  istat = cudaMemcpyAsync(y_in_dev, y_host, nxny * sizeof(y_host(1)), cudaMemcpyHostToDevice, stream2)

  threadsPerBlock = 256
  blocksPerGrid = (nxny + threadsPerBlock -1) / threadsPerBlock


  ! streams need to be synchronized before kernel launch - look into why
  istat = cudaStreamSynchronize(stream1)
  istat = cudaStreamSynchronize(stream2)

  ! This launches the actual kernel
  call transform_coordinates<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(x_in_dev, y_in_dev, x_out_dev, y_out_dev, nxny, x_obs, y_obs)
  
  ! Async transfer back to 'host' (main process on cpu)
  istat = cudaMemcpyAsync(x_transformed, x_out_dev, nxny * sizeof(x_transformed(1)), cudaMemcpyDeviceToHost, stream1)
  istat = cudaMemcpyAsync(y_transformed, y_out_dev, nxny * sizeof(y_transformed(1)), cudaMemcpyDeviceToHost, stream2)

  ! synchronize streams to make sure operations are complete
  istat = cudaStreamSynchronize(stream1)
  istat = cudaStreamSynchronize(stream2)

  deallocate(x_in_dev, y_in_dev, x_out_dev, y_out_dev)

  print *, 'Transformation complete. Sample output:'
  print *, 'Original point:', x_host(1), y_host(1)
  print *, 'Transformed point:', x_transformed(1), y_transformed(1)

  deallocate(x_host, y_host, x_transformed, y_transformed)

end program coordinate_transform