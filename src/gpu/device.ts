import { failure, required, timeOut } from "../utils.js"
import { BindGroupLayout, BindGroupLayouts, BindGroupLayoutDescriptor, BindGroupLayoutDescriptors } from "./group.js"
import { Buffer, SyncBuffer } from "./buffer.js"
import { Canvas } from "./canvas.js"
import { CommandEncoder } from "./encoder.js"
import { ShaderModule, ShaderModuleDescriptor, ShaderModuleDescriptors, ShaderModules } from "./shader.js"
import { Texture, Sampler } from "./texture.js"
import { PipelineLayout, PipelineLayoutDescriptor, PipelineLayoutDescriptors, PipelineLayouts } from "./pipeline.js"

export type DeviceDescriptor = {
    gpuDeviceDescriptor?: (adapter: GPUAdapter) => Promise<GPUDeviceDescriptor>
    xrCompatible?: boolean
}

export class Device {

    private destructionListeners: (() => void)[] = []
    private recoveryListeners: (() => void)[] = []

    constructor(private _wrapped: GPUDevice, private deviceDescriptor: DeviceDescriptor) {
        this._wrapped.lost.then(info => this.handleDeviceLoss(info))
    }

    private async handleDeviceLoss(info: GPUDeviceLostInfo) {
        const label = this._wrapped.label ?? 'unlabeled'
        if (info.reason === "destroyed") {
            console.info(`GPU Device '${label}' was lost because it was destroyed.`)
            this.destructionListeners.forEach(listener => listener())
        } else {
            console.warn(`GPU Device '${label}' was lost because:`, info.message, ". Attempting to recover ...")
            const { device, descriptor: _ } = await deviceAndDescriptor(this.deviceDescriptor)
            this._wrapped = device
            console.info(`GPU Device '${label}' was successfully recovered.`)
            this.recoveryListeners.forEach(listener => listener())
        }
    }

    get wrapped(): GPUDevice {
        return this._wrapped
    }
    
    /**
     * @deprecated Use `wrapped` instead.
     */
    get device(): GPUDevice {
        return this.wrapped
    }

    addDestructionListener(listener: () => void): () => void {
        return this.addListener(listener, this.destructionListeners)
    }

    addRecoveryListener(listener: () => void): () => void {
        return this.addListener(listener, this.recoveryListeners)
    }

    private addListener(listener: () => void, listeners: (() => void)[]) {
        const safeListener = () => {
            try {
                listener()
            } catch (e) {
                console.error("Error in device listener: ", e)
            }
        }
        listeners.push(safeListener)
        return () => listeners.splice(listeners.indexOf(safeListener), 1)
    }

    async shaderModules<D extends ShaderModuleDescriptors>(descriptors: D): Promise<ShaderModules<D>> {
        const result: Partial<ShaderModules<D>> = {}
        for (const k in descriptors) {
            result[k] = await this.shaderModule(k, descriptors[k])
        }
        return result as ShaderModules<D>
    }

    async shaderModule(label: string, descriptor: ShaderModuleDescriptor): Promise<ShaderModule> {
        return descriptor.path !== undefined
            ? await this.remoteShaderModule(label, descriptor.path, descriptor.templateFunction)
            : await this.inMemoryShaderModule(label, descriptor.code, descriptor.templateFunction);
    }

    async loadShaderModule(relativePath: string, templateFunction: (code: string) => string = s => s, basePath = "/shaders"): Promise<ShaderModule> {
        return await this.remoteShaderModule(relativePath, relativePath, templateFunction, basePath)
    }
    
    async remoteShaderModule(label: string, relativePath: string, templateFunction: (code: string) => string = s => s, basePath = "/shaders"): Promise<ShaderModule> {
        const response = await fetch(`${basePath}/${relativePath}`, { method : "get", mode : "no-cors" })
        const rawShaderCode = await response.text()
        return await this.inMemoryShaderModule(label, rawShaderCode, templateFunction)
    }

    async inMemoryShaderModule(label: string, rawShaderCode: string, templateFunction: (code: string) => string = s => s): Promise<ShaderModule> {
        const shaderCode = templateFunction(rawShaderCode)
        const shaderModule = new ShaderModule(label, this, shaderCode)

        if (await shaderModule.hasCompilationErrors()) {
            throw new Error("Module compilation failed!")
        }

        return shaderModule
    }

    enqueueCommands(name: string, ...encodings: ((encoder: CommandEncoder) => void)[]) {
        this.enqueue(...this.commands(name, ...encodings))
    }
    
    enqueueCommand(name: string, encoding: (encoder: CommandEncoder) => void) {
        this.enqueue(this.command(name, encoding))
    }

    enqueue(...commands: GPUCommandBuffer[]) {
        this.wrapped.queue.submit(commands)
    }
    
    commands(name: string, ...encodings: ((encoder: CommandEncoder) => void)[]): GPUCommandBuffer[] {
        return encodings.map((encoding, i) => this.command(`${name}#${i}`, encoding))
    }
    
    command(name: string, encoding: (encoder: CommandEncoder) => void): GPUCommandBuffer {
        const encoder = new CommandEncoder(name, this)
        encoding(encoder)
        return encoder.finish()
    }

    canvas(element: HTMLCanvasElement | string, sampleCount = 1): Canvas {
        return new Canvas(this, element, sampleCount)
    }

    texture(descriptor: GPUTextureDescriptor): Texture {
        return new Texture(this, descriptor)
    }

    sampler(descriptor: GPUSamplerDescriptor | undefined = undefined) {
        return new Sampler(this, descriptor)
    }

    buffer(label: string, usage: GPUBufferUsageFlags, dataOrSize: DataView | number, stride = 0): Buffer {
        return stride > 0 ? 
            new Buffer(label, this, usage, dataOrSize, stride) : 
            new Buffer(label, this, usage, dataOrSize) 
    }

    syncBuffer(label: string, usage: GPUBufferUsageFlags, dataOrSize: DataView | number, stride = 0): SyncBuffer {
        return stride > 0 ? 
            SyncBuffer.create(label, this, usage, dataOrSize, stride) : 
            SyncBuffer.create(label, this, usage, dataOrSize) 
    }

    groupLayouts<D extends BindGroupLayoutDescriptors>(descriptors: D): BindGroupLayouts<D> {
        const result: Partial<BindGroupLayouts<D>> = {}
        for (const k in descriptors) {
            result[k] = this.groupLayout(k, descriptors[k])
        }
        return result as BindGroupLayouts<D>
    }

    groupLayout<D extends BindGroupLayoutDescriptor>(label: string, descriptor: D): BindGroupLayout<D> {
        return new BindGroupLayout(this, label, descriptor)
    }

    pipelineLayouts<D extends PipelineLayoutDescriptors>(descriptors: D): PipelineLayouts<D> {
        const result: Partial<PipelineLayouts<D>> = {}
        for (const k in descriptors) {
            result[k] = this.pipelineLayout(k, descriptors[k])
        }
        return result as PipelineLayouts<D>
    }

    pipelineLayout<D extends PipelineLayoutDescriptor>(label: string, descriptor: D): PipelineLayout<D> {
        return new PipelineLayout(this, label, descriptor)
    }

    suggestedGroupSizes() {
        const limits = this.wrapped.limits
        const wgs = Math.max(
            limits.maxComputeWorkgroupSizeX,
            limits.maxComputeWorkgroupSizeY,
            limits.maxComputeWorkgroupSizeZ
        )
        const bitCount1D = Math.floor(Math.log2(wgs))
        const bitCount2DY = bitCount1D >>> 1
        const bitCount2DX = bitCount1D - bitCount2DY
        const bitCount3DZ = Math.floor(bitCount1D / 3)
        const bitCount3DY = (bitCount1D - bitCount3DZ) >>> 1
        const bitCount3DX = bitCount1D - bitCount2DY - bitCount3DZ
        const oneD = 1 << bitCount1D
        const twoDX = 1 << bitCount2DX
        const twoDY = 1 << bitCount2DY
        const threeDX = 1 << bitCount3DX
        const threeDY = 1 << bitCount3DY
        const threeDZ = 1 << bitCount3DZ
        return [ 
            [oneD], 
            [twoDX, twoDY ],
            [threeDX, threeDY, threeDZ]
        ]
    }
    

    async monitorErrors<T>(filter: GPUErrorFilter, expression: () => T): Promise<T> {
        this.wrapped.pushErrorScope(filter)
        const result = expression()
        const error = await this.wrapped.popErrorScope()
        if (error) {
            throw error
        }
        return result
    }    

    static async instance(deviceDescriptor: DeviceDescriptor = {}): Promise<Device> {
        var { device, descriptor } = await deviceAndDescriptor(deviceDescriptor)
        return new Device(device, {
            ...deviceDescriptor,
            gpuDeviceDescriptor: async () => descriptor,
        })
    }

}

async function deviceAndDescriptor(deviceDescriptor: DeviceDescriptor) {
    const gpu = required(navigator.gpu, () => "WebGPU is not supported in this environment!")
    const { gpuDeviceDescriptor, xrCompatible } = defaultedDescriptor(deviceDescriptor)
    const adapter = await requestAdapter(gpu, xrCompatible)
    var descriptor = await gpuDeviceDescriptor(adapter)
    const device = await requestDevice(adapter, descriptor)
    return { device, descriptor }
}

function defaultedDescriptor(descriptor: DeviceDescriptor): Required<DeviceDescriptor> {
    return { 
        gpuDeviceDescriptor: descriptor.gpuDeviceDescriptor ?? defaultGPUDeviceDescriptor,
        xrCompatible: descriptor.xrCompatible ?? false
    }
}

async function defaultGPUDeviceDescriptor(adapter: GPUAdapter): Promise<GPUDeviceDescriptor> {
    return adapter.info.isFallbackAdapter
        ? failure("The found GPU Adapter is a fallback one that may cause significant responsiveness problems!")
        : {} as GPUDeviceDescriptor
}

async function requestAdapter(gpu: GPU, xrCompatible: boolean) {
    const adapter = required(
        await timeOut(gpu.requestAdapter({ xrCompatible }), 5000, "GPU Adapter"),
        () => "No suitable GPU Adapter was found!"
    )
    console.debug("GPU Adapter Info:", adapter.info)
    console.debug("GPU Adapter Features:", [...adapter.features])
    console.debug("GPU Adapter Limits:", adapter.limits)
    return adapter
}

async function requestDevice(adapter: GPUAdapter, deviceDescriptor: GPUDeviceDescriptor) {
    const device = required(
        await timeOut(adapter.requestDevice(deviceDescriptor), 5000, "GPU Device"),
        () => "Failed to create a GPU Device!"
    )
    console.debug("GPU Device Features:", [...device.features])
    console.debug("GPU Device Limits:", device.limits)
    return device
}

