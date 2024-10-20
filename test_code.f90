! This code demonstrates strategies hiding data transfers via
! asynchronous data copies in multiple streams
module kernels_m
contains
    attributes(global) subroutine kernel(a, offset)
        implicit none
        real :: a(*)
        integer, value :: offset
        integer :: i
        real :: c, s, x
        i = offset + threadIdx%x + (blockIdx%x-1)*blockDim%x
        x = threadIdx%x + (blockIdx%x-1)*blockDim%x
        s = sin(x); c = cos(x)
        a(i) = a(i) + sqrt(s**2+c**2)
    end subroutine kernel
end module kernels_m

program testAsync
    use cudafor
    use kernels_m
    implicit none
    integer, parameter :: blockSize = 256, nStreams = 8
    integer, parameter :: n = 16*1024*blockSize*nStreams
    real, pinned, allocatable :: a(:)
    real, device :: a_d(n)
    integer(kind=cuda_Stream_Kind) :: stream(nStreams)
    type (cudaEvent) :: startEvent, stopEvent, dummyEvent
    real :: time
    integer :: i, istat, offset, streamSize = n/nStreams
    logical :: pinnedFlag
    type (cudaDeviceProp) :: prop
    istat = cudaGetDeviceProperties(prop, 0)
    write(*,"(' Device: ', a,/)") trim(prop%name)
    ! allocate pinned host memory
    allocate(a(n), STAT=istat, PINNED=pinnedFlag)
    if (istat /= 0) then
        write(*,*) 'Allocation of a failed'
        stop
    else
        if (.not. pinnedFlag) write(*,*) 'Pinned allocation failed'
        end if
    ! create events and streams
    istat = cudaEventCreate(startEvent)
    istat = cudaEventCreate(stopEvent)
    istat = cudaEventCreate(dummyEvent)
    do i = 1, nStreams
        istat = cudaStreamCreate(stream(i))
    enddo
    ! baseline case - sequential transfer and execute
    a = 0
    istat = cudaEventRecord(startEvent,0)
    a_d = a
    call kernel<<<n/blockSize, blockSize>>>(a_d, 0)
    a = a_d
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)
    write(*,*) 'Time for sequential transfer and execute (ms): ', time
    write(*,*) ' max error: ', maxval(abs(a-1.0))


    ! asynchronous version 1: loop over {copy, kernel, copy}
    a = 0
    istat = cudaEventRecord(startEvent,0)

    do i = 1, nStreams
        offset = (i-1)*streamSize
        istat = cudaMemcpyAsync(a_d(offset+1),a(offset+1),streamSize,stream(i))
        call kernel<<<streamSize/blockSize, blockSize, &
        0, stream(i)>>>(a_d,offset)
        istat = cudaMemcpyAsync(a(offset+1),a_d(offset+1),streamSize,stream(i))
    enddo
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)
    write(*,*) 'Time for asynchronous V1 transfer and execute (ms): ', time
    write(*,*) ' max error: ', maxval(abs(a-1.0))

    ! asynchronous version 2:
    ! loop over copy, loop over kernel, loop over copy
    a = 0
    istat = cudaEventRecord(startEvent,0)
    do i = 1, nStreams
        offset = (i-1)*streamSize
        istat = cudaMemcpyAsync(a_d(offset+1),a(offset+1),streamSize,stream(i))
    enddo
    do i = 1, nStreams
    offset = (i-1)*streamSize
    call kernel<<<streamSize/blockSize, blockSize, &
    0, stream(i)>>>(a_d,offset)
    enddo
    do i = 1, nStreams
    offset = (i-1)*streamSize
    istat = cudaMemcpyAsync(a(offset+1),a_d(offset+1),streamSize,stream(i))
    enddo
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)
    write(*,*) 'Time for asynchronous V2 transfer and execute (ms): ', time
    write(*,*) ' max error: ', maxval(abs(a-1.0))
    
    ! cleanup
    istat = cudaEventDestroy(startEvent)
    istat = cudaEventDestroy(stopEvent)
    istat = cudaEventDestroy(dummyEvent)
    do i = 1, nStreams
    istat = cudaStreamDestroy(stream(i))
    enddo
    deallocate(a)
end program testAsync