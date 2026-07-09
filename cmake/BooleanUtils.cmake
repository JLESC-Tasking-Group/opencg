macro(convert_bool var)
  if(${var})
      set(${var} 1)
  else()
      set(${var} 0)
  endif()
endmacro()

# cgiroption(NAME "DESCRIPTION" DEFAULT [VAR ...])
# Declare a boolean option NAME defaulting to DEFAULT.  The default is forced ON
# when any of the optional trailing variables (VAR ...) is defined, so pointing
# cmake at an install (e.g. -DLLVM_DIR=... / -DMLIR_DIR=...) auto-enables the
# matching feature and those variables are actually consumed.  An explicit
# -DNAME=... always wins, since option() keeps an already-set cache value.
macro(cgiroption NAME DESCRIPTION DEFAULT)
    set(_cgiroption_default "${DEFAULT}")
    foreach(_cgiroption_dep ${ARGN})
        if(DEFINED ${_cgiroption_dep})
            set(_cgiroption_default ON)
        endif()
    endforeach()
    option(${NAME} "${DESCRIPTION}" ${_cgiroption_default})
    convert_bool(${NAME})
endmacro()

