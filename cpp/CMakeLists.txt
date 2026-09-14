find_package(Threads REQUIRED)

add_library(
  lac_minorminer_core STATIC
  src/candidate_generator.cpp
  src/candidate_materialize.cpp
  src/candidate_metrics.cpp
  src/embedding_metrics.cpp
  src/embedding_state.cpp
  src/graph.cpp
  src/process_isolation.cpp
  src/search_session_lifecycle.cpp
  src/search_session_proposals.cpp
  src/weighted_router.cpp
)
set_target_properties(lac_minorminer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(lac_minorminer_core PUBLIC cxx_std_17)
target_link_libraries(lac_minorminer_core PUBLIC Threads::Threads)
if(MSVC)
  target_compile_options(lac_minorminer_core PRIVATE /W4)
else()
  target_compile_options(lac_minorminer_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_include_directories(
  lac_minorminer_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(LAC_MINORMINER_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(
    _core MODULE
    bindings/graph_bindings.cpp
    bindings/module.cpp
    bindings/process_isolation_bindings.cpp
    bindings/search_session_bindings.cpp
  )
  target_link_libraries(_core PRIVATE lac_minorminer_core)
  target_compile_definitions(_core PRIVATE LAC_MINORMINER_VERSION="${PROJECT_VERSION}")
  if(MSVC)
    target_compile_options(_core PRIVATE /W4)
  else()
    target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  install(TARGETS _core LIBRARY DESTINATION lac_minorminer RUNTIME DESTINATION lac_minorminer)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
