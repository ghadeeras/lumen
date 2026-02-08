import { Device } from "./device.js"
import { BindGroup, BindGroupLayout, BindGroupLayoutDescriptor } from "./group.js"
import { ShaderModule } from "./shader.js"
import { label } from "./utils.js"

export type PipelineEntries<D extends PipelineLayoutDescriptor> = {
    [k in keyof D["bindGroupLayouts"]]: D["bindGroupLayouts"][k]["layout"] extends BindGroupLayout<infer L> ? BindGroup<L> : never
}

export type PipelineLayouts<D extends PipelineLayoutDescriptors> = {
    [k in keyof D]: PipelineLayout<D[k]>
}
export type PipelineLayoutDescriptors = Record<string, PipelineLayoutDescriptor>
export type PipelineLayoutDescriptor = {
    label?: string
    bindGroupLayouts: Record<string, PipelineLayoutEntry<BindGroupLayoutDescriptor>>
}
export type PipelineLayoutEntry<L extends BindGroupLayoutDescriptor> = {
    group: number,
    layout: BindGroupLayout<L>
}
export class PipelineLayout<D extends PipelineLayoutDescriptor> {

    readonly wrapped: GPUPipelineLayout
    
    constructor(readonly device: Device, readonly descriptor: D) {
        const entries = descriptor.bindGroupLayouts
        const groups = Object.values(entries).map(g => g.group)
        const count = groups.length > 0 ? 1 + Math.max(...groups) : 0
        const bindGroupLayouts = new Array<GPUBindGroupLayout>(count)
        for (const k of Object.keys(entries)) {
            const entry = entries[k]
            bindGroupLayouts[entry.group] = entry.layout.wrapped
        }
        this.wrapped = device.wrapped.createPipelineLayout({
            label: descriptor.label,
            bindGroupLayouts
        })
    }

    computeInstance(module: ShaderModule, entryPoint: string): ComputePipeline<D> {
        return new ComputePipeline(this, module, entryPoint)
    }

    addTo(pass: GPUBindingCommandsMixin, groups: Partial<PipelineEntries<D>>) {
        for (const k of Object.keys(groups)) {
            const group = groups[k]
            if (group) {
                pass.setBindGroup(this.descriptor.bindGroupLayouts[k].group, group.wrapped)
            }
        }
    }

}

export class ComputePipeline<D extends PipelineLayoutDescriptor> {

    readonly wrapped: GPUComputePipeline
    readonly descriptor: GPUComputePipelineDescriptor

    constructor(readonly layout: PipelineLayout<D>, readonly module: ShaderModule, readonly entryPoint: string) {
        this.descriptor = {
            label: label(layout.descriptor.label, module.descriptor.label, entryPoint),
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

export function group<D extends BindGroupLayoutDescriptor>(group: number, layout: BindGroupLayout<D>): PipelineLayoutEntry<D> {
    return { group, layout }
}
