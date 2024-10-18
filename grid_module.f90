
module grid_module
  implicit none
  contains

  subroutine generate_grid(x, y, nx, ny, dx, dy)
    implicit none
    integer, intent(in) :: nx, ny
    real(8), intent(in) :: dx, dy
    real(8), intent(out) :: x(nx*ny), y(nx*ny)
    integer :: i, j, idx

    idx = 1
    do j = 0, ny - 1
      do i = 0, nx -1
        x(idx) = i * dx
        y(idx) = j * dy
        idx = idx + 1
      end do
    end do
  end subroutine generate_grid

end module grid_module