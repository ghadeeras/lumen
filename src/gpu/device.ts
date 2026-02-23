import * as utl from "../utils.js"
import * as pln from "./pipeline.js"
import * as shd from "./shader.js"
import * as grp from "./group.js"
import * as buf from "./buffer.js"
import * as txr from "./texture.js"
import { Canvas } from "./canvas.js"
import { CommandEncoder } from "./encoder.js"
import { withLabel } from "./utils.js"

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

    async shaderModules<D extends shd.ShaderModuleDescriptors>(descriptors: D, labelPrefix?: string): Promise<shd.ShaderModules<D>> {
        const result: Partial<shd.ShaderModules<D>> = {}
        for (const k in descriptors) {
            result[k] = await this.shaderModule(withLabel(descriptors[k], labelPrefix, k))
        }
        return result as shd.ShaderModules<D>
    }

    async shaderModule(descriptor: shd.ShaderModuleDescriptor): Promise<shd.ShaderModule> {
        return descriptor.path !== undefined
            ? await this.remoteShaderModule(descriptor.label ?? "shader", descriptor.path, descriptor.templateFunction)
            : await this.inMemoryShaderModule(descriptor.label ?? "shader", descriptor.code, descriptor.templateFunction);
    }

    async loadShaderModule(relativePath: string, templateFunction: (code: string) => string = s => s, basePath = "/shaders"): Promise<shd.ShaderModule> {
        return await this.remoteShaderModule(relativePath, relativePath, templateFunction, basePath)
    }
    
    async remoteShaderModule(label: string, relativePath: string, templateFunction: (code: string) => string = s => s, basePath = "/shaders"): Promise<shd.ShaderModule> {
        const response = await fetch(`${basePath}/${relativePath}`, { method : "get", mode : "no-cors" })
        const rawShaderCode = await response.text()
        return await this.inMemoryShaderModule(label, rawShaderCode, templateFunction)
    }

    async inMemoryShaderModule(label: string, rawShaderCode: string, templateFunction: (code: string) => string = s => s): Promise<shd.ShaderModule> {
        const shaderCode = templateFunction(rawShaderCode)
        const shaderModule = new shd.ShaderModule(label, this, shaderCode)

        if (await shaderModule.hasCompilationErrors()) {
            throw new Error("Module compilation failed!")
        }

        return shaderModule
    }

    enqueueCommands(label: string, encoding: (encoder: CommandEncoder) => void) {
        this.enqueue(this.commandBuffer(label, encoding))
    }

    enqueue(...commandBuffers: GPUCommandBuffer[]) {
        this.wrapped.queue.submit(commandBuffers)
    }
    
    commandBuffer(label: string, encoding: (encoder: CommandEncoder) => void): GPUCommandBuffer {
        const encoder = new CommandEncoder(label, this)
        encoding(encoder)
        return encoder.finish()
    }

    canvas(element: HTMLCanvasElement | string, sampleCount = 1): Canvas {
        return new Canvas(this, element, sampleCount)
    }

    texture(descriptor: GPUTextureDescriptor): txr.Texture {
        return new txr.Texture(this, descriptor)
    }

    sampler(descriptor: GPUSamplerDescriptor | undefined = undefined) {
        return new txr.Sampler(this, descriptor)
    }

    dataBuffers<D extends buf.DataBufferDescriptors>(descriptors: D, labelPrefix?: string): buf.DataBuffers<D> {
        return buf.DataBuffer.instances(this, descriptors, labelPrefix);
    }

    dataBuffer(descriptor: buf.DataBufferDescriptor): buf.DataBuffer {
        return buf.DataBuffer.instance(this, descriptor);
    }

    readBuffers<D extends buf.ReadBufferDescriptors>(descriptors: D, labelPrefix?: string): buf.ReadBuffers<D> {
        return buf.ReadBuffer.instances(this, descriptors, labelPrefix);
    }

    readBuffer(size: number, label?: string): buf.ReadBuffer {
        return buf.ReadBuffer.instance(this, size, label);
    }

    writeBuffers<D extends buf.WriteBufferDescriptors>(descriptors: D, labelPrefix?: string): buf.WriteBuffers<D> {
        return buf.WriteBuffer.instances(this, descriptors, labelPrefix);
    }

    writeBuffer(data: DataView, label?: string): buf.WriteBuffer {
        return buf.WriteBuffer.instance(this, data, label);
    }

    syncBuffers<D extends buf.DataBufferDescriptors>(descriptors: D, labelPrefix?: string): buf.SyncBuffers<D> {
        return buf.SyncBuffer.instances(this, descriptors, labelPrefix);
    }

    syncBuffer(descriptor: buf.DataBufferDescriptor): buf.SyncBuffer {
        return buf.SyncBuffer.instance(this, descriptor);
    }

    groupLayouts<D extends grp.BindGroupLayoutDescriptors>(descriptors: D, labelPrefix?: string): grp.BindGroupLayouts<D> {
        return grp.BindGroupLayout.instances(this, descriptors, labelPrefix)
    }

    groupLayout<D extends grp.BindGroupLayoutDescriptor>(descriptor: D, label?: string): grp.BindGroupLayout<D> {
        return grp.BindGroupLayout.instance(this, descriptor, label)
    }

    pipelineLayouts<D extends pln.PipelineLayoutDescriptors>(descriptors: D, labelPrefix?: string): pln.PipelineLayouts<D> {
        return pln.PipelineLayout.instances(this, descriptors, labelPrefix)
    }

    pipelineLayout<D extends pln.PipelineLayoutDescriptor>(descriptor: D, label?: string): pln.PipelineLayout<D> {
        return pln.PipelineLayout.instance(this, descriptor, label)
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
    const gpu = utl.required(navigator.gpu, () => "WebGPU is not supported in this environment!")
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
        ? utl.failure("The found GPU Adapter is a fallback one that may cause significant responsiveness problems!")
        : {} as GPUDeviceDescriptor
}

async function requestAdapter(gpu: GPU, xrCompatible: boolean) {
    const adapter = utl.required(
        await utl.timeOut(gpu.requestAdapter({ xrCompatible }), 5000, "GPU Adapter"),
        () => "No suitable GPU Adapter was found!"
    )
    console.debug("GPU Adapter Info:", adapter.info)
    console.debug("GPU Adapter Features:", [...adapter.features])
    console.debug("GPU Adapter Limits:", adapter.limits)
    return adapter
}

async function requestDevice(adapter: GPUAdapter, deviceDescriptor: GPUDeviceDescriptor) {
    const device = utl.required(
        await utl.timeOut(adapter.requestDevice(deviceDescriptor), 5000, "GPU Device"),
        () => "Failed to create a GPU Device!"
    )
    console.debug("GPU Device Features:", [...device.features])
    console.debug("GPU Device Limits:", device.limits)
    return device
}
