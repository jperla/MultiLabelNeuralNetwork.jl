import Msgpack: pack, unpack

typealias DictOrArray Union(Dict, Array)
typealias DictOrArrayOrNothing Union(Dict, Array, Nothing)

type MPFile
    # Message Packed file
    filename::UTF8String
    row_size::Int
    template::DictOrArray
end

MPFile(filename, row_size) = MPFile(filename, rowsize, nothing)

function new_message_pack_file(s::UTF8String, d::DictOrArray)
    @assert !isfile(s)
    packed = pack(d)
    mpf = new_message_pack_file(s, length(packed))
    append(mpf)
    return mpf
end

function new_message_pack_file(s::UTF8String, row_size::Int)
    @assert !isfile(s)
    mpf = MPFile(s, row_size)
    return mpf
end

function open_message_pack_file(s::UTF8String, row_size::Int)
    mpf = MPFile(s, row_size)
    return mpf
end

function append(mpf::MPFile, d)
    packed = pack(d)
    @assert length(packed) == mpf.row_size
    f = open(mpf.filename, "ab")
    write(f, packed)
    close(f)
end

function num_rows(mpf::MPFile)
    # TODO: optimize
    f = open(mpf.filename, "rb")
    d = length(read(f))
    close(f)
    @assert d % mpf.row_size == 0
    return d / mpf.row_size
end

function read(mpf::MPFile, i::Int)
    f = open(mpf.filename, "rb")
    seek(f, (i - 1) * mpf.row_size)
    binary_msg = read(f, mpf.row_size)
    close(f)
    return unpack(binary_msg)
end

