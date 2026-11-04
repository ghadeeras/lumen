import * as pln from "./pipeline.js"
import { Device } from "./device.js"

export class CommandEncoder {

    readonly wrapped: GPUCommandEncoder
    readonly descriptor: Readonly<GPUCommandEncoderDescriptor>

    constructor(label: string, readonly device: Device) {
        this.descriptor = { label }
        this.wrapped = this.device.wrapped.createCommandEncoder(this.descriptor)
    }

    finish(): GPUCommandBuffer {
        return this.wrapped.finish()
    }

    computePass<T>(label: string, passSetter: (pass: GPUComputePassEncoder) => T): T {
        const pass = this.wrapped.beginComputePass({ label })
        try {
            return passSetter(pass)
        } finally {
            pass.end()
        }
    }
    
    renderPass<T>(descriptor: GPURenderPassDescriptor, passSetter: (pass: GPURenderPassEncoder) => T): T {
        const pass = this.wrapped.beginRenderPass(descriptor)
        try {
            return passSetter(pass)
        } finally {
            pass.end()
        }
    }
    
}

export abstract class Pass<E extends GPUComputePassEncoder | GPURenderPassEncoder> {

    protected constructor(private device: Device, private label: string) {}

    enqueue() {
        this.device.enqueue(this.commandBuffer())
    }

    commandBuffer() {
        return this.device.commandBuffer(`${this.label} encoder`, encoder => this.encode(encoder)) 
    }

    encode(encoder: CommandEncoder) {
        this.pass(`${this.label} pass`, encoder, pass => this.inlineIn(pass))
        return encoder
    }

    protected abstract pass(label: string, encoder: CommandEncoder, encoding: (passEncoder: E) => void): void

    abstract inlineIn(passEncoder: E): E

}

export class ComputePass<D extends pln.PipelineLayoutDescriptor> extends Pass<GPUComputePassEncoder> implements ComputePassBuilder<D> {

    constructor(
        private pipeline: pln.ComputePipeline<D>, 
        private groups: Partial<pln.PipelineEntries<D>>, 
        private encoding: (passEncoder: GPUComputePassEncoder) => void = pass => pipeline.addTo(pass, groups)
    ) {
        super(pipeline.layout.device, pipeline.label + ".pass")
    }

    protected pass(label: string, encoder: CommandEncoder, encoding: (passEncoder: GPUComputePassEncoder) => void): void {
        encoder.computePass(label, encoding)
    }

    withGroups(groups: pln.PipelineEntries<D>): ComputePassBuilder<D> {
        return new ComputePass(this.pipeline, groups, 
            pass => this.pipeline.addGroupsTo(this.inlineIn(pass), groups)
        )
    }

    dispatchWorkGroups(workgroupCountX: number, workgroupCountY: number = 1, workgroupCountZ: number = 1): ComputePass<D> {
        return new ComputePass(this.pipeline, this.groups,
            pass => this.inlineIn(pass).dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ)
        )
    }

    dispatchWorkGroupsIndirect(indirectBuffer: GPUBuffer, indirectOffset: number): ComputePass<D> {
        return new ComputePass(this.pipeline, this.groups,
            pass => this.inlineIn(pass).dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset)
        )
    }

    inlineIn(passEncoder: GPUComputePassEncoder): GPUComputePassEncoder {
        this.encoding(passEncoder)
        return passEncoder
    }

}

export interface ComputePassBuilder<D extends pln.PipelineLayoutDescriptor> {
    
    withGroups(groups: pln.PipelineEntries<D>): ComputePassBuilder<D>
    
    dispatchWorkGroups(workgroupCountX: number, workgroupCountY?: number, workgroupCountZ?: number): ComputePass<D>

    dispatchWorkGroupsIndirect(indirectBuffer: GPUBuffer, indirectOffset: number): ComputePass<D>

}
