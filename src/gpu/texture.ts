import { Device } from "./device.js"
import { formatOf, Resource } from "./utils.js"

export class Texture implements Resource {

    private _wrapped: GPUTexture

    constructor(readonly device: Device, readonly descriptor: GPUTextureDescriptor) {
        this._wrapped = this.device.wrapped.createTexture(descriptor)
    }

    asBindingResource(): GPUBindingResource {
        return this._wrapped
    }

    get wrapped() {
        return this._wrapped
    }

    get label() {
        return this._wrapped.label
    }

    get size(): Required<Omit<GPUExtent3DDictStrict, "depth">> {
        return this._wrapped
    }

    set size(size: GPUExtent3DDictStrict) {
        this.descriptor.size = size
        this._wrapped.destroy()
        this._wrapped = this.device.wrapped.createTexture(this.descriptor)
    }

    isAnyOf(...flags: (keyof GPUTextureUsage)[]): boolean {
        return flags.reduce((b, f) => b || (this.descriptor.usage & GPUTextureUsage[f]) !== 0, false)
    }
    
    isAllOf(...flags: (keyof GPUTextureUsage)[]): boolean {
        return flags.reduce((b, f) => b && (this.descriptor.usage & GPUTextureUsage[f]) !== 0, true)
    }
    
    destroy() {
        this._wrapped.destroy()
    }

    createView(descriptor: GPUTextureViewDescriptor | undefined = undefined): TextureView {
        return new TextureView(this, descriptor)
    }

    depthState(state: Partial<GPUDepthStencilState> = {}): GPUDepthStencilState {
        return {
            ...state,
            format: state.format ?? formatOf(this.descriptor.format),
            depthCompare: state.depthCompare ?? "less",
            depthWriteEnabled: state.depthWriteEnabled ?? true
        }
    }

    asColorTargetState(state: Partial<GPUColorTargetState> = {}): GPUColorTargetState {
        return {
            ...state,
            format: this.descriptor.format
        }
    }
    
    async generateMipmaps() {
        if (this._wrapped.mipLevelCount == 1) {
            return
        }
        switch (this._wrapped.dimension) {
            case "2d": return await this.generate2DMipmaps() 
            default: throw "Unsupported dimension: " + this._wrapped.dimension
        }
    }
    
    private async generate2DMipmaps() {
        const code = /*wgsl*/`

            struct Vertex {
                @builtin(position) position: vec4<f32>,
                @location(0) clipPosition: vec2<f32>,
            };
            
            const triangle: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
                vec2(-1.0, -1.0),
                vec2( 3.0, -1.0),
                vec2(-1.0,  3.0),
            );
            
            @vertex
            fn v_main(@builtin(vertex_index) i: u32) -> Vertex {
                let clipPosition: vec2<f32> = triangle[i];
                return Vertex(vec4<f32>(clipPosition, 0.0, 1.0), clipPosition);
            }

            @group(0) @binding(0)
            var src_level: texture_2d<f32>;

            @group(0) @binding(1)
            var texture_sampler: sampler;

            @fragment
            fn f_main(vertex: Vertex) -> @location(0) vec4<f32> {
                let uv = vec2(0.5, -0.5) * (vertex.clipPosition + vec2(1.0, -1.0));
                return textureSample(src_level, texture_sampler, uv);
            }
        `
        const shader = await this.device.shaderModule({ label: "mipmap shader", code })
        const pipeline = await this.device.wrapped.createRenderPipelineAsync({
            label: "mipmap pipeline",
            layout: "auto",
            vertex: { module: shader.shaderModule },
            fragment: { 
                module: shader.shaderModule,
                targets: [{
                    format: this._wrapped.format
                }] 
            }
        })
        const sampler = this.device.sampler({
            label: "mipmap sampler",
            minFilter: "linear",
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge",
        })
        for (let layer = 0; layer < this._wrapped.depthOrArrayLayers; layer++) {
            const l = this._wrapped.textureBindingViewDimension === "2d-array" ? layer : undefined
            for (let level = 1; level < this._wrapped.mipLevelCount; level++) {
                const group = this.device.wrapped.createBindGroup({
                    label: "mipmap group",
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: this.mipLevel("2d", level - 1, l) }, 
                        { binding: 1, resource: sampler.sampler }
                    ]
                })
                this.device.enqueueCommands("mipmapping", encoder => {
                    encoder.renderPass({ 
                        colorAttachments: [{
                            storeOp: "store",
                            loadOp: "clear",
                            clearValue: { r: 0, g: 0, b: 0, a: 0 },
                            view: this.mipLevel("2d", level, l),
                        }] 
                    }, pass => {
                        pass.setPipeline(pipeline)
                        pass.setBindGroup(0, group)
                        pass.draw(3)
                    })
                })
            }
        }
    }

    private mipLevel<D extends GPUTextureDimension>(
        dimension: D, 
        level: number, 
        layer?: D extends "2d" ? number : never
    ): GPUTextureView {
        return this.createView({
            ...(layer !== undefined ? { baseArrayLayer: layer, arrayLayerCount: 1 } : {}),
            baseMipLevel: level,
            mipLevelCount: 1,
            dimension
        }).view
    }
}

export class TextureView implements Resource {

    readonly view: GPUTextureView
    
    constructor(readonly texture: Texture, descriptor: GPUTextureViewDescriptor | undefined = undefined) {
        this.view = texture.wrapped.createView(descriptor)
    }

    colorAttachment(clearValue: GPUColor | undefined = undefined): GPURenderPassColorAttachment {
        return {
            view: this.view,
            storeOp: clearValue === undefined || this.texture.isAnyOf("COPY_SRC", "TEXTURE_BINDING") ? "store" : "discard",
            loadOp: clearValue === undefined ? "load" : "clear",
            clearValue: clearValue,
        }
    }

    depthAttachment(clearValue: number | undefined = 1): GPURenderPassDepthStencilAttachment {
        return {
            view: this.view,
            depthStoreOp: clearValue === undefined || this.texture.isAnyOf("COPY_SRC", "TEXTURE_BINDING") ? "store" : "discard",
            depthLoadOp: clearValue === undefined ? "load" : "clear",
            depthClearValue: clearValue,
            stencilReadOnly: true,
        }
    }

    asBindingResource(): GPUBindingResource {
        return this.view
    }
    
}

export class Sampler implements Resource {

    readonly sampler: GPUSampler

    constructor(readonly device: Device, readonly descriptor: GPUSamplerDescriptor | undefined = undefined) {
        this.sampler = this.device.wrapped.createSampler(descriptor)
    }
   
    asBindingResource(): GPUBindingResource {
        return this.sampler
    }
    
}