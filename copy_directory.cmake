# Initially copy the workload directory.
if(NOT EXISTS ${destination_dir})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} ${destination_dir})
endif()
