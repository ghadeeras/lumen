export interface AddressSegment {

    offset?: number

    size?: number
    
}

export interface Element<T> extends AddressSegment {

    alignment: number

    size: number

    paddedSize: number

    offset: number

    stride: number

    packed: boolean

    view(dataOrSize?: T[] | number): DataView

    range(index?: number, count?: number): [number, number]

    segment(index?: number, count?: number): AddressSegment

    read(view: DataView, index?: number): T

    write(view: DataView, value: T, index?: number): void

    readMany(view: DataView, count: number, index?: number): T[]

    writeMany(view: DataView, values: T[], index?: number): void

    clone(offset: number, stride: number, packed: boolean): Element<T>

    times(length: number): StaticArray<T, this>
    
}

export type DataTypeOf<T extends Element<any>> = T extends Element<infer V> ? V : never

export type StructTypeOf<T  extends Record<string, Element<any>>> = {
    [K in keyof T]: DataTypeOf<T[K]>
}

abstract class BaseElement<T> implements Element<T> {
    
    readonly alignment: number
    readonly size: number
    readonly paddedSize: number
    readonly offset: number
    readonly stride: number
    readonly packed: boolean

    constructor(
        alignment: number,
        size: number,
        offset: number,
        stride: number,
        packed: boolean,
    ) {
        this.packed = packed
        this.alignment = packed ? 1 : alignment
        this.size = size
        this.paddedSize = alignTo(this.alignment, size)
        this.offset = alignTo(this.alignment, offset)
        this.stride = stride < this.paddedSize ? this.paddedSize : stride
    }

    view(dataOrSize: T[] | number = 1): DataView {
        if (typeof dataOrSize == 'number') {
            return this.emptyView(dataOrSize)
        } else {
            const result = this.emptyView(dataOrSize.length)
            this.writeMany(result, dataOrSize)
            return result
        }
    }

    private emptyView(size: number): DataView {
        return new DataView(new ArrayBuffer((size - 1) * this.stride + this.paddedSize))
    }

    segment(index = 0, count = 1): AddressSegment {
        const [l1, l2] = this.range(index, count)
        return { offset: l1, size: l2 - l1 }
    }

    range(index = 0, count = 1): [number, number] {
        const lo = this.offset + index * this.stride
        const hi = lo + (count - 1) * this.stride + this.paddedSize
        return [lo, Math.max(hi, lo)]
    }

    read(view: DataView, index = 0): T {
        return this.readOne(view, index)
    }

    write(view: DataView, value: T, index = 0): void {
        this.writeOne(view, value, index)
    }

    readMany(view: DataView, count: number, index: number = 0): T[] {
        const values: T[] = new Array(count)
        for (let i = 0; i < count; i++) {
            values[i] = this.readOne(view, index + i)
        }
        return values
    }

    writeMany(view: DataView, values: T[], index: number = 0): void {
        for (let i = 0; i < values.length; i++) {
            this.writeOne(view, values[i], index + i)
        }
    }

    times(length: number): StaticArray<T, this> {
        return new StaticArray(this, length, 0, 0, false)
    }

    protected abstract readOne(view: DataView, index: number): T

    protected abstract writeOne(view: DataView, value: T, index: number): void

    abstract clone(offset: number, stride: number, packed: boolean): Element<T>
    
}

export abstract class VertexElement<T> extends BaseElement<T> {

    constructor(
        readonly format: GPUVertexFormat,
        alignment: number,
        size: number,
        offset: number,
        stride: number,
        packed: boolean,
    ) {
        super(alignment, size, offset, stride, packed)
    }

    atLocation(location: number): GPUVertexAttribute {
        return {
            format: this.format,
            offset: this.offset,
            shaderLocation: location
        }
    }

}

abstract class Primitive32 extends VertexElement<number> {
    
    constructor(
        format: GPUVertexFormat,
        offset: number,
        stride: number,
        packed: boolean,
        private getter: (view: DataView, offset: number) => number, 
        private setter: (view: DataView, offset: number, value: number) => void
    ) {
        super(format, 4, 4, offset, stride, packed)
    }

    protected readOne(view: DataView, index: number): number {
        return this.getter(view, this.offset + this.stride * index)
    }

    protected writeOne(view: DataView, value: number, index: number): void {
        this.setter(view, this.offset + this.stride * index, value)
    }

    abstract clone(offset: number, stride: number, packed: boolean): Primitive32
    
    get x2(): Vec2<this> {
        return new Vec2(this, 0, 0, false)
    }
    
    get x3(): Vec3<this> {
        return new Vec3(this, 0, 0, false)
    }
    
    get x4(): Vec4<this> {
        return new Vec4(this, 0, 0, false)
    }
    
}

export class U32 extends Primitive32 {
    
    constructor(offset: number, stride: number, packed: boolean) {
        super("uint32", offset, stride, packed, 
            (view, offset) => view.getUint32(offset, true), 
            (view, offset, value) => view.setUint32(offset, value, true)
        )
    }

    clone(offset: number, stride: number, packed: boolean): U32 {
        return new U32(offset, stride, packed)
    }
    
}

export class I32 extends Primitive32 {
    
    constructor(offset: number, stride: number, packed: boolean) {
        super("sint32", offset, stride, packed, 
            (view, offset) => view.getInt32(offset, true), 
            (view, offset, value) => view.setInt32(offset, value, true)
        )
    }

    clone(offset: number, stride: number, packed: boolean): I32 {
        return new I32(offset, stride, packed)
    }
    
}

export class F32 extends Primitive32 {
    
    constructor(offset: number, stride: number, packed: boolean) {
        super("float32", offset, stride, packed, 
            (view, offset) => view.getFloat32(offset, true), 
            (view, offset, value) => view.setFloat32(offset, value, true)
        )
    }

    clone(offset: number, stride: number, packed: boolean): F32 {
        return new F32(offset, stride, packed)
    }
    
}

export abstract class VecN<C extends number[]> extends VertexElement<C> {

    constructor(format: GPUVertexFormat, alignment: number, size: number, offset: number, stride: number, packed: boolean) {
        super(format, alignment, size, offset, stride, packed)
    }

    get x2(): Mat2<C, this> {
        return new Mat2(this, 0, 0, false)
    }
    
    get x3(): Mat3<C, this> {
        return new Mat3(this, 0, 0, false)
    }
    
    get x4(): Mat4<C, this> {
        return new Mat4(this, 0, 0, false)
    }
    
}

export class Vec2<T extends Primitive32> extends VecN<[number, number]> {

    readonly x: T
    readonly y: T

    constructor(readonly component: T, offset: number, stride: number, packed: boolean) {
        super("float32x2", 8, 8, offset, stride, packed)
        this.x = clone(component, this.offset, this.stride, packed)
        this.y = clone(component, this.x.offset + this.x.paddedSize, this.stride, packed)
    }
    
    protected readOne(view: DataView, index: number): [number, number] {
        return [
            this.x.read(view, index),
            this.y.read(view, index),
        ]
    }

    protected writeOne(view: DataView, value: [number, number], index: number): void {
        this.x.write(view, value[0], index)
        this.y.write(view, value[1], index)
    }

    clone(offset: number, stride: number, packed: boolean): Vec2<T> {
        return new Vec2(this.component, offset, stride, packed)
    }

}

export class Vec3<T extends Primitive32> extends VecN<[number, number, number]> {

    readonly x: T
    readonly y: T
    readonly z: T

    constructor(readonly component: T, offset: number, stride: number, packed: boolean) {
        super("float32x3", 16, 12, offset, stride, packed)
        this.x = clone(component, this.offset, this.stride, packed)
        this.y = clone(component, this.x.offset + this.x.paddedSize, this.stride, packed)
        this.z = clone(component, this.y.offset + this.y.paddedSize, this.stride, packed)
    }
    
    protected readOne(view: DataView, index: number): [number, number, number] {
        return [
            this.x.read(view, index),
            this.y.read(view, index),
            this.z.read(view, index),
        ]
    }

    protected writeOne(view: DataView, value: [number, number, number], index: number): void {
        this.x.write(view, value[0], index)
        this.y.write(view, value[1], index)
        this.z.write(view, value[2], index)
    }

    clone(offset: number, stride: number, packed: boolean): Vec3<T> {
        return new Vec3(this.component, offset, stride, packed)
    }

}

export class Vec4<T extends Primitive32> extends VecN<[number, number, number, number]> {

    readonly x: T
    readonly y: T
    readonly z: T
    readonly w: T

    constructor(readonly component: T, offset: number, stride: number, packed: boolean) {
        super("float32x4", 16, 16, offset, stride, packed)
        this.x = clone(component, this.offset, this.stride, packed)
        this.y = clone(component, this.x.offset + this.x.paddedSize, this.stride, packed)
        this.z = clone(component, this.y.offset + this.y.paddedSize, this.stride, packed)
        this.w = clone(component, this.z.offset + this.z.paddedSize, this.stride, packed)
    }
    
    protected readOne(view: DataView, index: number): [number, number, number, number] {
        return [
            this.x.read(view, index),
            this.y.read(view, index),
            this.z.read(view, index),
            this.w.read(view, index),
        ]
    }

    protected writeOne(view: DataView, value: [number, number, number, number], index: number): void {
        this.x.write(view, value[0], index)
        this.y.write(view, value[1], index)
        this.z.write(view, value[2], index)
        this.w.write(view, value[3], index)
    }

    clone(offset: number, stride: number, packed: boolean): Vec4<T> {
        return new Vec4(this.component, offset, stride, packed)
    }

}

export class Mat2<T extends number[], V extends VecN<T>> extends BaseElement<[T, T]> {

    readonly x: V
    readonly y: V

    constructor(readonly component: V, offset: number, stride: number, packed: boolean) {
        super(component.alignment, arraySize(component, 2, packed), offset, stride, packed)
        this.x = clone(component, this.offset, this.stride, packed)
        this.y = clone(component, this.x.offset + this.x.paddedSize, this.stride, packed)
    }
    
    protected readOne(view: DataView, index: number): [T, T] {
        return [
            this.x.read(view, index),
            this.y.read(view, index),
        ]
    }

    protected writeOne(view: DataView, value: [T, T], index: number): void {
        this.x.write(view, value[0], index)
        this.y.write(view, value[1], index)
    }

    clone(offset: number, stride: number, packed: boolean): Mat2<T, V> {
        return new Mat2(this.component, offset, stride, packed)
    }

}

export class Mat3<T extends number[], V extends VecN<T>> extends BaseElement<[T, T, T]> {

    readonly x: V
    readonly y: V
    readonly z: V

    constructor(readonly component: V, offset: number, stride: number, packed: boolean) {
        super(component.alignment, arraySize(component, 3, packed), offset, stride, packed)
        this.x = clone(component, this.offset, this.stride, packed)
        this.y = clone(component, this.x.offset + this.x.paddedSize, this.stride, packed)
        this.z = clone(component, this.y.offset + this.y.paddedSize, this.stride, packed)
    }
    
    protected readOne(view: DataView, index: number): [T, T, T] {
        return [
            this.x.read(view, index),
            this.y.read(view, index),
            this.z.read(view, index),
        ]
    }

    protected writeOne(view: DataView, value: [T, T, T], index: number): void {
        this.x.write(view, value[0], index)
        this.y.write(view, value[1], index)
        this.z.write(view, value[2], index)
    }

    clone(offset: number, stride: number, packed: boolean): Mat3<T, V> {
        return new Mat3(this.component, offset, stride, packed)
    }

}

export class Mat4<T extends number[], V extends VecN<T>> extends BaseElement<[T, T, T, T]> {

    readonly x: V
    readonly y: V
    readonly z: V
    readonly w: V

    constructor(readonly component: V, offset: number, stride: number, packed: boolean) {
        super(component.alignment, arraySize(component, 4, packed), offset, stride, packed)
        this.x = clone(component, this.offset, this.stride, packed)
        this.y = clone(component, this.x.offset + this.x.paddedSize, this.stride, packed)
        this.z = clone(component, this.y.offset + this.y.paddedSize, this.stride, packed)
        this.w = clone(component, this.z.offset + this.z.paddedSize, this.stride, packed)
    }
    
    protected readOne(view: DataView, index: number): [T, T, T, T] {
        return [
            this.x.read(view, index),
            this.y.read(view, index),
            this.z.read(view, index),
            this.w.read(view, index),
        ]
    }

    protected writeOne(view: DataView, value: [T, T, T, T], index: number): void {
        this.x.write(view, value[0], index)
        this.y.write(view, value[1], index)
        this.z.write(view, value[2], index)
        this.w.write(view, value[3], index)
    }

    clone(offset: number, stride: number, packed: boolean): Mat4<T, V> {
        return new Mat4(this.component, offset, stride, packed)
    }

}

export class StaticArray<T, I extends Element<T>> extends BaseElement<T[]> {

    readonly items: I[]

    constructor(readonly item: I, length: number, offset: number, stride: number, packed: boolean) {
        super(item.alignment, arraySize(item, length, packed), offset, stride, packed)
        this.items = cloneItems<T, I>(item, length, this.offset, this.stride, packed)
    }
    
    protected readOne(view: DataView, index: number): T[] {
        const result: T[] = new Array<T>(this.items.length)
        for (let i = 0; i < this.items.length; i++) {
            result[i] = this.items[i].read(view, index)
        }
        return result
    }

    protected writeOne(view: DataView, value: T[], index: number): void {
        for (let i = 0; i < this.items.length; i++) {
            this.items[i].write(view, value[i], index)
        }
    }

    clone(offset: number, stride: number, packed: boolean): StaticArray<T, I> {
        return new StaticArray(this.item, this.items.length, offset, stride, packed)
    }

}

export class Struct<T extends Record<string, Element<any>>> extends BaseElement<StructTypeOf<T>> {

    readonly members: T
    
    constructor(readonly membersOrder: (keyof T)[], members: T, offset: number, stride: number, packed: boolean) {
        super(structAlignment(membersOrder, members), structSize(membersOrder, members, packed), offset, stride, packed)
        this.members = cloneStruct(membersOrder, members, this.offset, this.stride, packed)
    }

    protected readOne(view: DataView, index: number): StructTypeOf<T> {
        const result: Partial<StructTypeOf<T>> = {}
        for (const key of this.membersOrder) {
            result[key] = this.members[key].read(view, index)
        }
        return result as StructTypeOf<T>
    }

    protected writeOne(view: DataView, value: StructTypeOf<T>, index: number): void {
        for (const key of this.membersOrder) {
            this.members[key].write(view, value[key], index)
        }
    }

    clone(offset: number, stride: number, packed: boolean): Struct<T> {
        return new Struct(this.membersOrder, this.members, offset, stride, packed)
    }

    asVertex(attributes: (keyof T)[] = this.membersOrder): Vertex<T> {
        return new Vertex(attributes, this)
    }

    asObject(getter: <A>(element: Element<A>) => A, setter: <A>(element: Element<A>, value: A) => void): StructTypeOf<T> {
        const result: Partial<StructTypeOf<T>> = {}
        for (const key of this.membersOrder) {
            const member = this.members[key]
            if (member instanceof Struct) {
                const memberValue = member.asObject(getter, setter)
                Object.defineProperty(result, key, {
                    get() { return memberValue },
                    set(value: any) { return setter(member, value) }
                })
            } else if (member instanceof StaticArray && member.item instanceof Struct) {
                const memberValue = (member as StaticArray<any, Struct<any>>).items.map(i => i.asObject(getter, setter))
                Object.defineProperty(result, key, {
                    get() { return memberValue },
                    set(value: any) { return setter(member, value) }
                })
            } else {
                Object.defineProperty(result, key, {
                    get() { return getter(member) },
                    set(value: any) { return setter(member, value) }
                })
            }
        }
        return result as StructTypeOf<T>
    }

}

export class Vertex<T extends Record<string, Element<any>>> {

    constructor(readonly attributes: (keyof T)[], readonly struct: Struct<T>) {
    }

    asBufferLayout(stepMode: GPUVertexStepMode = "vertex", baseIndex = 0): GPUVertexBufferLayout {
        return {
            arrayStride: this.struct.stride,
            attributes: this.attributes.map((name, index) => {
                const member = this.struct.members[name]
                if (!(member instanceof VertexElement)) {
                    throw new Error('Only vertex element can be attributes of a vertex.')
                }
                return member.atLocation(baseIndex + index)
            }),
            stepMode: stepMode
        }
    }

    sub(attributes: (keyof T)[] = this.attributes): Vertex<T> {
        return this.struct.asVertex(attributes)
    }

}

function arraySize<T, I extends Element<T>>(item: I, length: number, packed: boolean): number {
    return (packed ? item.size : item.paddedSize) * length
}

function cloneItems<T, I extends Element<T>>(item: I, length: number, offset: number, stride: number, packed: boolean) {
    const items = new Array<I>(length)
    let itemOffset = offset
    for (let i = 0; i < length; i++) {
        const clonedItem = clone(item, itemOffset, stride, packed)
        items[i] = clonedItem
        itemOffset = clonedItem.offset + clonedItem.paddedSize
    }
    return items
}

function structAlignment<T extends Record<string, Element<any>>>(membersOrder: (keyof T)[], struct: T): number {
    let result = 1
    for (const key of membersOrder) {
        const member = struct[key]
        if (member.alignment > result) {
            result = member.alignment
        }
    }
    return result
}

function structSize<T extends Record<string, Element<any>>>(membersOrder: (keyof T)[], struct: T, packed: boolean): number {
    let result = 0
    for (const key of membersOrder) {
        const member = struct[key]
        result = packed ? result : alignTo(member.alignment, result)
        result += member.size    
    }
    const alignment = packed ? 1 : structAlignment(membersOrder, struct)
    return alignTo(alignment, result)
}

function cloneStruct<T extends Record<string, Element<any>>>(membersOrder: (keyof T)[], struct: T, offset: number, stride: number, packed: boolean): T {
    const result: Partial<T> = {}
    for (const key of membersOrder) {
        const t = clone(struct[key], offset, stride, packed)
        result[key] = t
        offset = t.offset + t.size
    }
    return result as T
}

function clone<T extends Element<any>>(type: T, offset: number, stride: number, packed: boolean): T {
    const result = type.clone(offset, stride, packed)
    if (type.constructor !== result.constructor) {
        throw new Error("Clone is not same class as original!")
    }
    return result as T
}

export const u32 = new U32(0, 0, false)
export const i32 = new I32(0, 0, false)
export const f32 = new F32(0, 0, false)

export function vec2<T extends Primitive32>(component: T): Vec2<T> {
    return component.x2
}

export function vec3<T extends Primitive32>(component: T): Vec3<T> {
    return component.x3
}

export function vec4<T extends Primitive32>(component: T): Vec4<T> {
    return component.x4
}

export const mat2x2 = vec2(f32).x2
export const mat2x3 = vec3(f32).x2
export const mat2x4 = vec4(f32).x2
export const mat3x2 = vec2(f32).x3
export const mat3x3 = vec3(f32).x3
export const mat3x4 = vec4(f32).x3
export const mat4x2 = vec2(f32).x4
export const mat4x3 = vec3(f32).x4
export const mat4x4 = vec4(f32).x4

export function struct<T extends Record<string, Element<any>>>(members: T, membersOrder: (keyof T)[] = Object.keys(members)): Struct<T> {
    return new Struct(membersOrder, members, 0, 0, false)
}

export function packed<T extends Record<string, VertexElement<any>>>(membersOrder: (keyof T)[], members: T): Struct<T> {
    return new Struct(membersOrder, members, 0, 0, true)
}

export function vertex<T extends Record<string, VertexElement<any>>>(members: T, membersOrder: (keyof T)[] = Object.keys(members)): Vertex<T> {
    return new Vertex(membersOrder, packed<T>(membersOrder, members))
}

function alignTo(alignment: number, size: number) {
    return Math.ceil(size / alignment) * alignment
}
