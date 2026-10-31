import { DataBuffer } from "./buffer.js";
import { Canvas } from "./canvas.js";
import { Sampler, Texture } from "./texture.js";

export interface Resource<R extends DataBuffer | Texture | Sampler> {

    label: string

    asBindingResource(): GPUBindingResource

    baseResource(): R

}

export type TextureFormatSource =
    GPUColorTargetState |
    GPUTextureFormat |
    Texture |
    Canvas

export type TypedArray = 
    Float32Array |
    Int32Array |
    Int16Array |
    Int8Array |
    Uint32Array |
    Uint16Array |
    Uint8Array

export type Only<T, K extends keyof T> = {
    [k in K]: T[k] 
} & {
    [k in Exclude<keyof T, K>]?: never
}
export type IfSet<T, K extends keyof T, R> = T[K] extends (object | any[] | string | number | boolean | symbol) ? R : never
export type Redefine<T, K extends keyof T, V> = {
    [k in keyof T]: k extends K ? V : T[k]
}
export type ReplaceValues<T extends Record<string, any>, V> = {
    [k in keyof T]: V;
};
export type StrictExclude<T, S extends T> = Exclude<T, S>
export type Supplier<T> = () => T

type Ref<T> = {
    value?: T
}

export function asColorTargetState(formatted: TextureFormatSource): GPUColorTargetState {
    return typeof formatted != 'string' && ("blend" in formatted || "writeMask" in formatted) 
        ? formatted 
        :  { format: formatOf(formatted) }
}

export function formatOf(formatted: TextureFormatSource): GPUTextureFormat {
    return typeof formatted !== 'string' ?
        formatted instanceof Texture ?
            formatted.descriptor.format :
            formatted.format :
        formatted
}

export function dataView(array: TypedArray): DataView {
    return new DataView(array.buffer, array.byteOffset, array.byteLength)
}

export function float32Array(view: DataView): Float32Array {
    return new Float32Array(view.buffer, view.byteOffset, view.byteLength / Float32Array.BYTES_PER_ELEMENT)
}

export function int32Array(view: DataView): Int32Array {
    return new Int32Array(view.buffer, view.byteOffset, view.byteLength / Int32Array.BYTES_PER_ELEMENT)
}

export function int16Array(view: DataView): Int16Array {
    return new Int16Array(view.buffer, view.byteOffset, view.byteLength / Int16Array.BYTES_PER_ELEMENT)
}

export function uint16Array(view: DataView): Uint16Array {
    return new Uint16Array(view.buffer, view.byteOffset, view.byteLength / Uint16Array.BYTES_PER_ELEMENT)
}

export function int8Array(view: DataView): Int8Array {
    return new Int8Array(view.buffer, view.byteOffset, view.byteLength / Int8Array.BYTES_PER_ELEMENT)
}

export function uint8Array(view: DataView): Uint8Array {
    return new Uint8Array(view.buffer, view.byteOffset, view.byteLength / Uint8Array.BYTES_PER_ELEMENT)
}

export function withLabel<D extends GPUObjectDescriptorBase>(descriptor: D, ...labelParts: (string | undefined)[]) {
    return {
        label: label(...labelParts),
        ...descriptor
    }
}

export function label(...parts: (string | undefined)[]) {
    return parts.reduce((a, p) => a !== undefined 
        ? p !== undefined 
            ? `${a}.${p}` 
            : a 
        : p !== undefined 
            ? p 
            : undefined, 
        undefined
    )
}

export function replaceValues<R extends Record<string, any>, B>(
    record: R,
    replace: (value: R[keyof R], key: string & (keyof R)) => B
): ReplaceValues<R, B> {
    const result: Partial<ReplaceValues<R, B>> = {}
    for (const key in record) {
        result[key] = replace(record[key], key)
    }
    return result as ReplaceValues<R, B>
}

export function failure<T>(message: string): T {
    throw new Error(message)
}

export function required<T>(value: T | null | undefined, message: (v: null | undefined) => string = v => `Required value is ${value}!`): T {
    return value === null || value === undefined 
        ? failure(message(value as (null | undefined))) 
        : value
}

export function lazily<T>(constructor: Supplier<T>): Supplier<T> {
    const ref: Ref<T> = {}
    return () => ref.value === undefined ?
        ref.value = constructor() :
        ref.value
}

export function values<K extends string | number | symbol, V>(record: Record<K, V>): V[] {
    const result: V[] = []
    for (const key in record) {
        result.push(record[key])
    }
    return result
}

export function later(): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, 0))
}

export function timeOut<T>(promise: Promise<T>, timeInMilliseconds: number, tag: string): Promise<T> {
    return new Promise((resolve, reject) => {
        const id: [number | null] = [setTimeout(() => {
            id[0] = null
            reject(new Error(`[${tag}] Timed out after ${timeInMilliseconds} milliseconds!`))
        }, timeInMilliseconds)]

        promise
            .then(value => {
                if (id[0] !== null) {
                    clearTimeout(id[0])
                    resolve(value)
                }
            })
            .catch(reason => {
                if (id[0] !== null) {
                    clearTimeout(id[0])
                    reject(reason)
                }
            })
    })
}
