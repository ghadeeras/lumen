import { Device } from "./device.js";
import { Element } from "./types.js";
import { PipelineLayoutEntry } from "./index.js";
import * as bfr from "./buffer.js";
import * as txr from "./texture.js";
import * as utl from "./utils.js";

export type CompatibleBindGroup<L extends BindGroupLayout> = BindGroup<InferBindGroupLayoutDescriptor<L>>
export type CompatibleBindGroups<L extends BindGroupLayout> = BindGroups<InferBindGroupLayoutDescriptor<L>>

export type BindGroups<L extends BindGroupLayoutDescriptor = {}, G extends BindGroupDescriptors<L> = {}> = {
    [K in keyof G]: BindGroup<L>
};

export interface BindGroup<L extends BindGroupLayoutDescriptor = {}> {
    readonly label: string 
    readonly device: Device
    readonly layout: BindGroupLayout<L>
    readonly entries: BindGroupDescriptor<L>
    readonly wrapped: GPUBindGroup
}
class BindGroupImpl<L extends BindGroupLayoutDescriptor = {}> implements BindGroup<L> {

    readonly device: Device
    readonly wrapped: GPUBindGroup

    constructor(readonly layout: BindGroupLayout<L>, readonly label: string, readonly entries: BindGroupDescriptor<L>) {
        this.device = layout.device;
        this.wrapped = layout.device.wrapped.createBindGroup({
            label: `group ${layout.label} / ${label}`,
            layout: layout.wrapped,
            entries: Object.keys(entries).map(key =>({
                binding: layout.entries[key].binding,
                resource: entries[key].asBindingResource()
            }))
        })
    }
    
}

export type CompatibleBindGroupDescriptor<L extends BindGroupLayout> = BindGroupDescriptor<InferBindGroupLayoutDescriptor<L>>
export type CompatibleBindGroupDescriptors<L extends BindGroupLayout> = BindGroupDescriptors<InferBindGroupLayoutDescriptor<L>>
export type InferBindGroupLayoutDescriptor<L extends BindGroupLayout> = L extends BindGroupLayout<infer D> ? D : never

export type BindGroupDescriptors<L extends BindGroupLayoutDescriptor = {}> = Record<string, BindGroupDescriptor<L>> 
export type BindGroupDescriptor<L extends BindGroupLayoutDescriptor = {}> = { 
    [K in keyof L]: InferResourceFromBindGroupLayoutEntry<L[K]> 
}
export type InferResourceFromBindGroupLayoutEntry<T extends BindGroupLayoutEntry> =
    T extends BindGroupLayoutEntry<infer R> ? R : never

export type BindGroupLayouts<D extends BindGroupLayoutDescriptors = {}> = {
    [K in keyof D]: BindGroupLayout<D[K]>
};

export class BindGroupLayout<L extends BindGroupLayoutDescriptor = {}> {

    readonly wrapped: GPUBindGroupLayout

    constructor(readonly device: Device, readonly label: string, readonly entries: L) {
        this.wrapped = device.wrapped.createBindGroupLayout({
            label,
            entries: Object.values(entries).map(entry => entry.wrapped)
        })
    }

    asEntry(group: number): PipelineLayoutEntry<L> {
        return {
            group,
            layout: this
        }
    }

    bindGroups<G extends BindGroupDescriptors<L>>(descriptors: G, labelPrefix?: string): BindGroups<L, G> {
        const bindGroups: Partial<BindGroups<L, G>> = {};
        for (const key in descriptors) {
            bindGroups[key] = this.bindGroup(descriptors[key], utl.label(labelPrefix,  key));
        }
        return bindGroups as BindGroups<L, G>;
    }

    bindGroup(entries: BindGroupDescriptor<L>, label?: string): BindGroup<L> {
        return new BindGroupImpl(this, label ?? this.labelFrom(entries), entries)
    }

    private labelFrom(entries: BindGroupDescriptor<L>): string {
        return `[ ${Object.keys(entries)
            .sort((a, b) => this.entries[a].binding - this.entries[b].binding)
            .map(key => `${key}: ${entries[key].label}`)
            .join(", ")
        } ]`;
    }

    static instances<D extends BindGroupLayoutDescriptors>(device: Device, descriptors: D, labelPrefix?: string): BindGroupLayouts<D> {
        const layouts: Partial<BindGroupLayouts<D>> = {};
        for (const key in descriptors) {
            layouts[key] = BindGroupLayout.instance(device, descriptors[key], utl.label(labelPrefix, key));
        }
        return layouts as BindGroupLayouts<D>;
    }

    static instance<D extends BindGroupLayoutDescriptor>(device: Device, entries: D, label?: string): BindGroupLayout<D> {
        return new BindGroupLayout(device, label ?? BindGroupLayout.labelFrom(entries), entries)
    }

    private static labelFrom<D extends BindGroupLayoutDescriptor>(entries: D): string {
        return `[ ${Object.keys(entries)
            .sort((a, b) => entries[a].binding - entries[b].binding)
            .map(key => `${key}: ${entries[key].resource.constructor.name}`)
            .join(", ")
        } ]`;
    }

}

export type BindGroupLayoutDescriptors = Record<string, BindGroupLayoutDescriptor>;
export type BindGroupLayoutDescriptor = Record<string, BindGroupLayoutEntry>;

export class BindGroupLayoutEntry<R extends utl.Resource = utl.Resource> {

    readonly wrapped: GPUBindGroupLayoutEntry

    constructor(readonly binding: number, readonly visibility: number, readonly resource: ResourceBindingLayout<R>) {
        this.wrapped = {
            binding,
            visibility,
            ...this.resource.wrapped
        }
    }

}

export function uniform<T>(contentType: Element<T>, hasDynamicOffset?: boolean): BufferBindingLayout<T> {
    return bufferBindingLayout<T>("uniform", contentType, hasDynamicOffset)
}

export function storage<T>(accessMode: AccessMode, contentType: Element<T>, hasDynamicOffset?: boolean): BufferBindingLayout<T> {
    return bufferBindingLayout<T>(accessMode != "read" ? "storage" : "read-only-storage", contentType, hasDynamicOffset)
}

function bufferBindingLayout<T>(type: GPUBufferBindingType, contentType: Element<T>, hasDynamicOffset: boolean | undefined): BufferBindingLayout<T> {
    return new BufferBindingLayout({
        type,
        minBindingSize: contentType.size,
        hasDynamicOffset
    }, contentType);
}

export function texture_storage_1d(format: GPUTextureFormat, accessMode: AccessMode = "write"): TextureStorageBindingLayout {
    return texture_storage("1d", format, accessMode)
}

export function texture_storage_2d(format: GPUTextureFormat, accessMode: AccessMode = "write"): TextureStorageBindingLayout {
    return texture_storage("2d", format, accessMode)
}

export function texture_storage_2d_array(format: GPUTextureFormat, accessMode: AccessMode = "write"): TextureStorageBindingLayout {
    return texture_storage("2d-array", format, accessMode)
}

export function texture_storage_3d(format: GPUTextureFormat, accessMode: AccessMode = "write"): TextureStorageBindingLayout {
    return texture_storage("3d", format, accessMode)
}

function texture_storage(viewDimension: GPUTextureViewDimension, format: GPUTextureFormat, accessMode: AccessMode): TextureStorageBindingLayout {
    return new TextureStorageBindingLayout({
        viewDimension,
        format,
        access: accessMode == "read" ? "read-only" : accessMode == "write" ? "write-only" : "read-write",
    });
}

export function texture_1d(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("1d", sampleType)
}

export function texture_2d(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("2d", sampleType)
}

export function texture_2d_array(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("2d-array", sampleType)
}

export function texture_cube(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("cube", sampleType)
}

export function texture_cube_array(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("cube-array", sampleType)
}

export function texture_3d(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("3d", sampleType)
}

export function texture_depth_2d(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("2d", sampleType)
}

export function texture_depth_2d_array(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("2d-array", sampleType)
}

export function texture_depth_cube(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("cube", sampleType)
}

export function texture_depth_cube_array(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("cube-array", sampleType)
}

export function texture_multisampled_2d(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("2d", sampleType, true)
}

export function texture_depth_multisampled_2d(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("2d", sampleType, true)
}

export function texture_external(sampleType: GPUTextureSampleType): TextureBindingLayout {
    return texture("2d", sampleType)
}

function texture(viewDimension: GPUTextureViewDimension, sampleType: GPUTextureSampleType, multisampled?: boolean): TextureBindingLayout {
    return new TextureBindingLayout({ viewDimension, sampleType, multisampled });
}

export function sampler(type: "filtering" | "non-filtering" = "filtering"): SamplerBindingLayout {
    return new SamplerBindingLayout({ type });
}

export function sampler_comparison(): SamplerBindingLayout {
    return new SamplerBindingLayout({ type: "comparison" });
}

export type AccessMode = "read" | "write" | "read_write";

export abstract class ResourceBindingLayout<R extends utl.Resource = utl.Resource> {

    constructor(readonly wrapped: Omit<GPUBindGroupLayoutEntry, "binding" | "visibility">) {}

    asEntry(binding: number, stage: keyof typeof GPUShaderStage, ...otherStages: (keyof typeof GPUShaderStage)[]): BindGroupLayoutEntry<R> {
        const visibility = otherStages.reduce((v, s) => v | GPUShaderStage[s], GPUShaderStage[stage]);
        return new BindGroupLayoutEntry(binding, visibility, this);
    }

}

export class BufferBindingLayout<T> extends ResourceBindingLayout<bfr.DataBuffer | bfr.SyncBuffer> {

    constructor(readonly buffer: GPUBufferBindingLayout, readonly contentType: Element<T>) {
        super({ buffer })
    }

}

export class TextureStorageBindingLayout extends ResourceBindingLayout<txr.Texture | txr.TextureView> {

    constructor(readonly storageTexture: GPUStorageTextureBindingLayout) {
        super({ storageTexture })
    }

}

export class TextureBindingLayout extends ResourceBindingLayout<txr.Texture | txr.TextureView> {

    constructor(readonly texture: GPUTextureBindingLayout) {
        super({ texture })
    }

}

export class SamplerBindingLayout extends ResourceBindingLayout<txr.Sampler> {

    constructor(readonly sampler: GPUSamplerBindingLayout) {
        super({ sampler })
    }

}
