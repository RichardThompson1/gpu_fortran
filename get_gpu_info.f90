
program cufinfo
    use cudafor
    integer istat, num, numdevices
    type(cudadeviceprop) :: prop
    istat = cudaGetDeviceCount(numdevices)
    do num = 0, numdevices-1
        istat = cudaGetDeviceProperties(prop, num)
        call printDeviceProperties(prop, num)
    end do
end program cufinfo
    !
subroutine printDeviceProperties(prop, num)
    use cudafor
    type(cudadeviceprop) :: prop
    integer num
    ilen = verify(prop%name, ' ', .true.)
    write (*,900) "Device Number: " ,num
    write (*,901) "Device Name: " ,prop%name(1:ilen)
    write (*,903) "Total Global Memory: ",real(prop%totalGlobalMem)/1e9," Gbytes"
    write (*,902) "sharedMemPerBlock: " ,prop%sharedMemPerBlock," bytes"
    write (*,900) "regsPerBlock: " ,prop%regsPerBlock
    write (*,900) "warpSize: " ,prop%warpSize
    write (*,900) "maxThreadsPerBlock: " ,prop%maxThreadsPerBlock
    write (*,904) "maxThreadsDim: " ,prop%maxThreadsDim
    write (*,904) "maxGridSize: " ,prop%maxGridSize
    write (*,903) "ClockRate: " ,real(prop%clockRate)/1e6," GHz"
    write (*,902) "Total Const Memory: " ,prop%totalConstMem," bytes"
    write (*,905) "Compute Capability Revision: ",prop%major,prop%minor
    write (*,902) "TextureAlignment: " ,prop%textureAlignment," bytes"
    write (*,906) "deviceOverlap: " ,prop%deviceOverlap
    write (*,900) "multiProcessorCount: ",prop%multiProcessorCount
    write (*,906) "integrated: " ,prop%integrated
    write (*,906) "canMapHostMemory: " ,prop%canMapHostMemory
    write (*,906) "ECCEnabled: " ,prop%ECCEnabled
    write (*,906) "UnifiedAddressing: " ,prop%unifiedAddressing
    write (*,900) "L2 Cache Size: " ,prop%l2CacheSize
    write (*,900) "maxThreadsPerSMP: " ,prop%maxThreadsPerMultiProcessor
    900 format (a,i0)
    901 format (a,a)
    902 format (a,i0,a)
    903 format (a,f5.3,a)
    904 format (a,2(i0,1x,'x',1x),i0)
    905 format (a,i0,'.',i0)
    906 format (a,l0)
    return
end subroutine printDeviceProperties
