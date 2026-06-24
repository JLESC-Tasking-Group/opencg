macro(convert_bool var)
  if(${var})
      set(${var} 1)
  else()
      set(${var} 0)
  endif()
endmacro()

macro(cgiroption NAME DESCRIPTION DEFAULT)
    option(${NAME} "${DESCRIPTION}" ${DEFAULT})
    convert_bool(${NAME})
endmacro()

