# Writes a C++ header holding the bytes of a file (used to compile the chat page into klein.exe).
# embed_file(<input> <output header> <symbol>)
function(embed_file input output symbol)
    file(READ "${input}" hex HEX)
    string(LENGTH "${hex}" hex_len)
    math(EXPR n "${hex_len} / 2")
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
    string(REGEX REPLACE "(0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,)" "\\1\n" bytes "${bytes}")
    file(WRITE "${output}" "// generated from ${input} - do not edit\n#pragma once\n#include <cstddef>\n"
        "static const unsigned char ${symbol}[] = {\n${bytes}0x00};\n"
        "static const size_t ${symbol}_len = ${n};\n")
endfunction()
