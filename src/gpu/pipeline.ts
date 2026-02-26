import * as grp from "./group.js"
import * as utl from "./utils.js"
import { Device } from "./device.js"
import { ShaderModule } from "./shader.js"
import { ComputePassBuilder } from "./index.js"

export type CompatiblePipelineEntries<L extends PipelineLayout> = PipelineEntries<InferPipelineLayoutDescriptor<L>>
export type InferPipelineLayoutDescriptor<L extends PipelineLayout> = L extends PipelineLayout<infer D> ? D : never

export type PipelineLayouts<D extends PipelineLayoutDescriptors> = {
    [k in keyof D]: PipelineLayout<D[k]>
}
export class PipelineLayout<D extends PipelineLayoutDescriptor = {}> {

    readonly wrapped: GPUPipelineLayout
    
    private constructor(readonly device: Device, readonly label: string, readonly descriptor: D) {
        const groups = Object.values(descriptor).map(g => g.group)
        const count = groups.length > 0 ? 1 + Math.max(...groups) : 0
        const bindGroupLayouts = new Array<GPUBindGroupLayout>(count)
        for (const k of Object.keys(descriptor)) {
            const entry = descriptor[k]
            bindGroupLayouts[entry.group] = entry.layout.wrapped
        }
        this.wrapped = device.wrapped.createPipelineLayout({ label, bindGroupLayouts })
    }

    async computePipeline(compute: ProgrammableStage, label?: string): Promise<ComputePipeline<D>> {
        return await ComputePipeline.instance({ label, layout: this, compute })
    }

    bindGroups<G extends keyof D>(name: G, groups: grp.CompatibleBindGroupDescriptors<D[G]["layout"]>) {
        return this.descriptor[name].layout.bindGroups(groups)
    }

    bindGroup<G extends keyof D>(name: G, group: grp.CompatibleBindGroupDescriptor<D[G]["layout"]>) {
        return this.descriptor[name].layout.bindGroup(group)
    }

    addTo(pass: GPUBindingCommandsMixin, groups: PipelineEntries<D>) {
        for (const k of Object.keys(groups)) {
            const group = groups[k]
            if (group) {
                pass.setBindGroup(this.descriptor[k].group, group.wrapped)
            }
        }
    }

    static instances<D extends PipelineLayoutDescriptors>(device: Device, descriptors: D, labelPrefix?: string): PipelineLayouts<D> {
        const result: Partial<PipelineLayouts<D>> = {}
        for (const key in descriptors) {
            result[key] = PipelineLayout.instance(device, descriptors[key], utl.label(labelPrefix, key))
        }
        return result as PipelineLayouts<D>
    }

    static instance<D extends PipelineLayoutDescriptor>(device: Device, descriptor: D, label?: string): PipelineLayout<D> {
        return new PipelineLayout(device, label ?? PipelineLayout.labelFrom(descriptor), descriptor)
    }
    
    static labelFrom<D extends PipelineLayoutDescriptor>(descriptor: D): string {
        return `[ ${Object.keys(descriptor)
            .sort((a, b) => descriptor[a].group - descriptor[b].group)
            .map(key => `${key}: ${descriptor[key].layout.label}`)
            .join(", ")
        } ]`;
    }

}

export type PipelineLayoutDescriptors = Record<string, PipelineLayoutDescriptor>
export type PipelineLayoutDescriptor = Record<string, PipelineLayoutEntry<grp.BindGroupLayoutDescriptor>>
export type PipelineLayoutEntry<L extends grp.BindGroupLayoutDescriptor> = {
    group: number,
    layout: grp.BindGroupLayout<L>
}
export type PipelineEntries<D extends PipelineLayoutDescriptor> = Partial<{
    [k in keyof D]: D[k]["layout"] extends grp.BindGroupLayout<infer L> ? grp.BindGroup<L> : never
}>

export class ComputePipeline<D extends PipelineLayoutDescriptor> {

    private _label: string

    constructor(readonly wrapped: GPUComputePipeline, readonly descriptor: ComputePipelineDescriptor<D>) {
        this._label = descriptor.label ?? "compute pipeline"
    }

    get label() {
        return this._label
    }

    get layout() {
        return this.descriptor.layout
    }

    bindGroups<G extends keyof D>(name: G, groups: grp.CompatibleBindGroupDescriptors<D[G]["layout"]>) {
        return this.layout.bindGroups(name, groups)
    }

    bindGroup<G extends keyof D>(name: G, group: grp.CompatibleBindGroupDescriptor<D[G]["layout"]>) {
        return this.layout.bindGroup(name, group)
    }

    addTo(pass: GPUComputePassEncoder, groups: PipelineEntries<D> = {}): GPUComputePassEncoder {
        pass.setPipeline(this.wrapped)
        this.addGroupsTo(pass, groups)
        return pass
    }

    addGroupsTo(pass: GPUBindingCommandsMixin, groups: PipelineEntries<D>) {
        this.layout.addTo(pass, groups)
    }

    withGroups(groups: PipelineEntries<D>): ComputePassBuilder<D> {
        return new ComputePassBuilder(this, groups)
    }

    static async instance<D extends PipelineLayoutDescriptor>(descriptor: ComputePipelineDescriptor<D>): Promise<ComputePipeline<D>> {
        const gpuDescriptor = utl.withLabel<GPUComputePipelineDescriptor>({
            label: descriptor.label,
            layout: descriptor.layout.wrapped, 
            compute: {
                entryPoint: descriptor.compute.entryPoint,
                module: descriptor.compute.module.wrapped,
            }
        }, descriptor.layout.label, descriptor.compute.module.label, descriptor.compute.entryPoint)
        const wrapped = await descriptor.layout.device.wrapped.createComputePipelineAsync(gpuDescriptor)
        return new ComputePipeline(wrapped, descriptor)
    }

}

export type ComputePipelineDescriptor<D extends PipelineLayoutDescriptor> = {
    layout: PipelineLayout<D>
    label?: string | undefined
    compute: ProgrammableStage
}
export type ProgrammableStage = utl.Redefine<GPUProgrammableStage, "module", ShaderModule>

export function group<D extends grp.BindGroupLayoutDescriptor>(group: number, layout: grp.BindGroupLayout<D>): PipelineLayoutEntry<D> {
    return { group, layout }
}
