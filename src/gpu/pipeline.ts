import { Device } from "./device.js"
import { ShaderModule } from "./shader.js"
import { label } from "./utils.js"
import * as grp from "./group.js"

export type PipelineEntries<D extends PipelineLayoutDescriptor> = {
    [k in keyof D]: D[k]["layout"] extends grp.BindGroupLayout<infer L> ? grp.BindGroup<L> : never
}

export type PipelineLayouts<D extends PipelineLayoutDescriptors> = {
    [k in keyof D]: PipelineLayout<D[k]>
}
export type PipelineLayoutDescriptors = Record<string, PipelineLayoutDescriptor>
export type PipelineLayoutDescriptor = Record<string, PipelineLayoutEntry<grp.BindGroupLayoutDescriptor>>
export type PipelineLayoutEntry<L extends grp.BindGroupLayoutDescriptor> = {
    group: number,
    layout: grp.BindGroupLayout<L>
}
export class PipelineLayout<D extends PipelineLayoutDescriptor> {

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

    computePipeline(module: ShaderModule, entryPoint: string): ComputePipeline<D> {
        return new ComputePipeline(this, module, entryPoint)
    }

    addTo(pass: GPUBindingCommandsMixin, groups: Partial<PipelineEntries<D>>) {
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
            result[key] = PipelineLayout.instance(device, descriptors[key], label(labelPrefix, key))
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

export class ComputePipeline<D extends PipelineLayoutDescriptor> {

    readonly wrapped: GPUComputePipeline
    readonly descriptor: GPUComputePipelineDescriptor

    constructor(readonly layout: PipelineLayout<D>, readonly module: ShaderModule, readonly entryPoint: string) {
        this.descriptor = {
            label: label(layout.label, module.descriptor.label, entryPoint),
            layout: layout.wrapped, 
            compute: {
                entryPoint,
                module: module.shaderModule,
            }
        }
        this.wrapped = layout.device.wrapped.createComputePipeline(this.descriptor)
    }

    addTo(pass: GPUComputePassEncoder, groups: Partial<PipelineEntries<D>> = {}) {
        pass.setPipeline(this.wrapped)
        this.addGroupsTo(pass, groups)
    }

    addGroupsTo(pass: GPUBindingCommandsMixin, groups: Partial<PipelineEntries<D>>) {
        this.layout.addTo(pass, groups)
    }

}

export function group<D extends grp.BindGroupLayoutDescriptor>(group: number, layout: grp.BindGroupLayout<D>): PipelineLayoutEntry<D> {
    return { group, layout }
}
