import { Device } from "./device.js"
import { ComputePipeline, PipelineEntries, PipelineLayoutDescriptor } from "./index.js"

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

    computePass<T>(passSetter: (pass: GPUComputePassEncoder) => T): T {
        const pass = this.wrapped.beginComputePass()
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

export class ComputePass extends Pass<GPUComputePassEncoder> {

    constructor(device: Device, label: string, private encoding: (passEncoder: GPUComputePassEncoder) => void) {
        super(device, label)
    }

    protected pass(label: string, encoder: CommandEncoder, encoding: (passEncoder: GPUComputePassEncoder) => void): void {
        encoder.computePass(encoding)
    }

    inlineIn(passEncoder: GPUComputePassEncoder): GPUComputePassEncoder {
        this.encoding(passEncoder)
        return passEncoder
    }

}

export class ComputePassBuilder<D extends PipelineLayoutDescriptor> {
    
    constructor(
        private pipeline: ComputePipeline<D>, 
        private groups: Partial<PipelineEntries<D>>
    ) {}

    dispatchWorkGroups(workgroupCountX: number, workgroupCountY?: number, workgroupCountZ?: number): ComputePass {
        return new ComputePass(this.pipeline.layout.device, this.pipeline.label, pass => this.init(pass)
            .dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ))
    }

    dispatchWorkGroupsIndirect(indirectBuffer: GPUBuffer, indirectOffset: number): ComputePass {
        return new ComputePass(this.pipeline.layout.device, this.pipeline.label, pass => this.init(pass)
            .dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset))
    }

    private init(pass: GPUComputePassEncoder) {
        return this.pipeline.addTo(pass, this.groups)
    }

}
