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

type Ref<T> = {
    value?: T
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