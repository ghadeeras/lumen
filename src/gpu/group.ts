import { Only, IfSet, Redefine } from "../utils.js";
import { DataBuffer, SyncBuffer } from "./buffer.js";
import { Device } from "./device.js";
import { PipelineLayoutEntry } from "./pipeline.js";
import { Sampler, Texture, TextureView } from "./texture.js";

/*
 * Bind Group types
 */
export type BindGroups<L extends BindGroupLayoutDescriptor, D extends BindGroupDescriptors<L>> = {
    [k in keyof D]: BindGroup<L>
} 
export type BindGroupDescriptors<D extends BindGroupLayoutDescriptor> = Record<string, BindGroupDescriptor<D>>;
export type BindGroupDescriptor<D extends BindGroupLayoutDescriptor> = {
    entries: BindGroupEntries<D>
}
export type BindGroupEntries<D extends BindGroupLayoutDescriptor> = {
    [k in keyof D["entries"]]: BindGroupResource<D["entries"][k]>
}
export type BindGroupResource<T extends BindGroupLayoutEntry> = 
      IfSet<T, "buffer", SyncBuffer | DataBuffer> 
    | IfSet<T, "texture", TextureView | Texture> 
    | IfSet<T, "storageTexture", TextureView | Texture> 
    | IfSet<T, "externalTexture", TextureView | Texture> 
    | IfSet<T, "sampler", Sampler>

/*
 * Bind Group "Layout" types
 */
export type BindGroupLayouts<D extends BindGroupLayoutDescriptors> = {
    [k in keyof D]: BindGroupLayout<D[k]>
}
export type BindGroupLayoutDescriptors = Record<string, BindGroupLayoutDescriptor>;
export type BindGroupLayoutDescriptor = {
    entries: Record<string, BindGroupLayoutEntry>
}
export type BindGroupLayoutEntry = 
      BindGroupResourceLayout<"buffer">
    | BindGroupResourceLayout<"texture">
    | BindGroupResourceLayout<"storageTexture">
    | BindGroupResourceLayout<"externalTexture">
    | BindGroupResourceLayout<"sampler">
export type BindGroupResourceLayout<T extends ResourceType> = Only<
    Required<Redefine<GPUBindGroupLayoutEntry, "visibility", (keyof typeof GPUShaderStage)[]>>, 
    T | BindingAttributes
>
export type BindingAttributes = Exclude<keyof GPUBindGroupLayoutEntry, ResourceType>
export type ResourceType = "buffer" | "texture" | "storageTexture" | "externalTexture" | "sampler"

export class BindGroupLayout<D extends BindGroupLayoutDescriptor> {

    readonly wrapped: GPUBindGroupLayout

    constructor(readonly device: Device, readonly label: string, readonly descriptor: D) {
        const entryList: GPUBindGroupLayoutEntry[] = [];
        for (const key of Object.keys(descriptor.entries)) {
            const entry = descriptor.entries[key]
            const newEntry = {
                ...entry,
                visibility: entry.visibility.map(s => GPUShaderStage[s]).reduce((a, b) => a | b, 0)
            }
            entryList.push(newEntry)
        }
        this.wrapped = device.wrapped.createBindGroupLayout({
            label,
            entries: entryList
        })
    }

    instance(label: string, descriptor: BindGroupDescriptor<D>): BindGroup<D> {
        return new BindGroup(label, this, descriptor)
    }

    instances<G extends BindGroupDescriptors<D>>(descriptors: G): BindGroups<D, G> {
        const result: Partial<BindGroups<D, G>> = {}
        for (const key in descriptors) {
            result[key] = this.instance(key, descriptors[key])
        }
        return result as BindGroups<D, G>
    }

    asEntry(group: number): PipelineLayoutEntry<D> {
        return { layout: this, group }
    }

}

export class BindGroup<D extends BindGroupLayoutDescriptor> {

    readonly wrapped: GPUBindGroup

    constructor(readonly label: string, readonly layout: BindGroupLayout<D>, readonly descriptor: BindGroupDescriptor<D>) {
        const entryList: GPUBindGroupEntry[] = [];
        for (const key of Object.keys(descriptor.entries)) {
            entryList.push({
                binding: layout.descriptor.entries[key].binding,
                resource: descriptor.entries[key].asBindingResource()
            })
        }
        this.wrapped = layout.device.wrapped.createBindGroup({
            label: `${layout.label}@${label}`,
            layout: layout.wrapped,
            entries: entryList
        })
    }

}