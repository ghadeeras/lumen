import * as utl from "./utils.js";
import { Device } from "./device.js";

export type ShaderModules<D extends ShaderModuleDescriptors> = {
    [k in keyof D]: ShaderModule
}
export class ShaderModule {

    readonly wrapped: GPUShaderModule
    readonly descriptor: Readonly<GPUShaderModuleDescriptor>

    constructor(readonly label: string, readonly device: Device, code: string) {
        this.descriptor = { code, label };
        this.wrapped = this.device.wrapped.createShaderModule(this.descriptor)
        if (this.wrapped === null) {
            throw new Error("Module compilation failed!")
        }
    }

    async hasCompilationErrors() {
        if (!this.wrapped.getCompilationInfo) {
            // TODO remove check when compilationInfo becomes supported in all browsers. 
            return false
        }
        const info = await this.wrapped.getCompilationInfo()
        for (const message of info.messages) {
            switch (message.type) {
                case "info": console.log(message); break
                case "warning": console.warn(message); break
                case "error": console.error(message); break
                default:
            }
        }
        return info.messages.some(m => m.type == "error")
    }

    computePipeline(entryPoint?: string, ) {
        return this.device.wrapped.createComputePipeline({
            compute: { 
                module: this.wrapped,
                entryPoint: entryPoint, 
            },
            layout: "auto",
            label: `${this.wrapped.label}/${entryPoint}`
        })
    }

    vertexState(entryPoint: string, buffers: (GPUVertexBufferLayout | number)[]): GPUVertexState {
        const index = [0]
        return {
            module: this.wrapped,
            entryPoint: entryPoint,
            buffers: buffers.map(buffer => {
                if (typeof buffer == 'number') {
                    index[0] += buffer
                    return null
                }
                return {
                    ...buffer,
                    attributes: [...buffer.attributes].map(attribute => ({
                        ...attribute,
                        shaderLocation: index[0]++ 
                    }))
                }
            })
        }
    }

    fragmentState(entryPoint: string, targets: (utl.TextureFormatSource | null)[]): GPUFragmentState {
        return {
            module: this.wrapped,
            entryPoint: entryPoint,
            targets: targets.map(target => target !== null 
                ? utl.asColorTargetState(target)
                : null
            )
        }
    }

    static async instances<D extends ShaderModuleDescriptors>(device: Device, descriptors: D, labelPrefix?: string): Promise<ShaderModules<D>> {
        const result: Partial<ShaderModules<D>> = {}
        const tuplePromises = Object.entries(descriptors).map(async ([k, d]) => ({ 
            key: k, 
            module: await ShaderModule.instance(device, utl.withLabel(d, labelPrefix, k)) 
        }))
        const tuples = await Promise.all(tuplePromises)
        for (const tuple of tuples) {
            result[tuple.key as keyof typeof descriptors] = tuple.module
        }
        return result as ShaderModules<D>
    }

    static async instance(device: Device, descriptor: ShaderModuleDescriptor): Promise<ShaderModule> {
        return descriptor.path !== undefined
            ? await remoteShaderModule(device, descriptor.label ?? "shader", descriptor.path, descriptor.templateFunction)
            : await inMemoryShaderModule(device, descriptor.label ?? "shader", descriptor.code, descriptor.templateFunction);
    }

}

export type ShaderModuleDescriptors = Record<string, ShaderModuleDescriptor>
export type ShaderModuleDescriptor = ShaderModuleCode & {
    label?: string,
    compilationHints?: Array<GPUShaderModuleCompilationHint>;
    templateFunction?: (code: string) => string
}
export type ShaderModuleCode = utl.Only<ShaderModuleCodeAttributes, "path"> | utl.Only<ShaderModuleCodeAttributes, "code">
export type ShaderModuleCodeAttributes = {
    path: string
    code: string
}

async function remoteShaderModule(device: Device, label: string, relativePath: string, templateFunction: (code: string) => string = s => s, basePath = ""): Promise<ShaderModule> {
    const response = await fetch(`${basePath}/${relativePath}`, { method : "get", mode : "no-cors" })
    const rawShaderCode = await response.text()
    return await inMemoryShaderModule(device, label, rawShaderCode, templateFunction)
}

async function inMemoryShaderModule(device: Device, label: string, rawShaderCode: string, templateFunction: (code: string) => string = s => s): Promise<ShaderModule> {
    const shaderCode = templateFunction(rawShaderCode)
    const shaderModule = new ShaderModule(label, device, shaderCode)

    if (await shaderModule.hasCompilationErrors()) {
        throw new Error("Module compilation failed!")
    }

    return shaderModule
}
