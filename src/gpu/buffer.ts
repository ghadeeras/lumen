import { ReplaceValues, StrictExclude } from "../utils.js";
import { Device } from "./device.js";
import { DataSegment, Element } from "./types.js";
import { label, Resource, withLabel } from "./utils.js";

export type SyncBuffers<D extends DataBufferDescriptors> = ReplaceValues<D, SyncBuffer>;

export class SyncBuffer implements Resource {

    private dirtyRange: [number, number]

    private constructor(readonly gpuBuffer: DataBuffer, readonly cpuBuffer: DataView) {
        this.dirtyRange = [cpuBuffer.byteLength, 0]
    }

    get label() {
        return this.gpuBuffer.label;
    }

    asBindingResource(binding: Omit<GPUBufferBinding, "buffer"> = {}): GPUBindingResource {
        return this.gpuBuffer.asBindingResource(binding)
    }

    get<T>(element: Element<T>): T {
        return element.read(this.cpuBuffer)
    }

    set<T>(element: Element<T>, value: T) {
        element.write(this.cpuBuffer, value)
        this.dirty(element.range())
    }

    private dirty(range: [number, number]) {
        if (this.dirtyRange[0] > this.dirtyRange[1]) {
            setTimeout(() => this.clean())
        }
        if (range[0] < this.dirtyRange[0]) {
            this.dirtyRange[0] = range[0]
        }
        if (range[1] > this.dirtyRange[1]) {
            this.dirtyRange[1] = range[1]
        }
    }

    private clean() {
        const offset = this.dirtyRange[0];
        const size = this.dirtyRange[1] - this.dirtyRange[0];
        this.gpuBuffer.set({ offset, size }).fromData(this.cpuBuffer, offset);
        this.dirtyRange[0] = this.cpuBuffer.byteLength
        this.dirtyRange[1] = 0
    }

    static instances<D extends DataBufferDescriptors>(device: Device, descriptors: D, labelPrefix?: string): SyncBuffers<D> {
        const result: Partial<SyncBuffers<D>> = {};
        for (const k in descriptors) {
            result[k] = SyncBuffer.instance(device, withLabel(descriptors[k], labelPrefix, k));
        }
        return result as SyncBuffers<D>;
    }

    static instance(device: Device, descriptor: DataBufferDescriptor) {
        const gpuBuffer = device.dataBuffer(descriptor);
        const cpuBuffer = descriptor.data === undefined ? new DataView(new ArrayBuffer(gpuBuffer.size)) : descriptor.data;
        return new SyncBuffer(gpuBuffer, cpuBuffer);
    }

}

export interface BaseBuffer extends Resource {
    wrapped: GPUBuffer;
    size: number;
    label: string;
    device: Device;
    destroy(): void;
    asBindingResource(binding?: Omit<GPUBufferBinding, "buffer"> | null): GPUBindingResource;
}

export abstract class AbstractBuffer implements BaseBuffer {

    protected constructor(readonly device: Device, private _wrapped: GPUBuffer) {
    }

    get wrapped() {
        return this._wrapped;
    }

    set wrapped(b) {
        this._wrapped.destroy();
        this._wrapped = b;
    }

    get size() {
        return this._wrapped.size;
    }

    get label() {
        return this._wrapped.label;
    }

    destroy() {
        this._wrapped.destroy();
    }

    asBindingResource(binding: Omit<GPUBufferBinding, "buffer"> | null = null) {
        return binding === null ? this._wrapped : {
            ...binding,
            buffer: this._wrapped,
        };
    }

}

export interface SourceBuffer extends BaseBuffer {
    copy(segment?: DataSegment): DestinationClause<void>;
    set(segment?: DataSegment): BufferWriter;
}

export interface DestinationBuffer extends BaseBuffer {
    copy(segment?: DataSegment): SourceClause<void>;
    get(segment?: DataSegment): BufferReader;
}

export class DataBuffer extends AbstractBuffer implements SourceBuffer, DestinationBuffer {

    private constructor(device: Device, readonly descriptor: DataBufferDescriptor) {
        super(device, descriptor.size !== undefined ?
            newBlankBuffer(device, usageFlags(descriptor), descriptor.size, descriptor.label) :
            newInitializedBuffer(device, usageFlags(descriptor), descriptor.data, descriptor.label)
        );
        this.descriptor = descriptor;
    }

    copy(segment: DataSegment = {}): DestinationClause<void> & SourceClause<void> {
        return {
            ...copy(segment.size).from(this, segment.offset),
            ...copy(segment.size).to(this, segment.offset),
        };
    }

    async setData(data: DataView) {
        if (data.byteLength > this.size) {
            this.wrapped = newInitializedBuffer(this.device, usageFlags(this.descriptor), data, this.label);
        } else {
            this.set({ offset: 0, size: data.byteLength }).fromData(data);
        }
    }

    set(bufferSegment: DataSegment = {}): BufferWriter {
        return {
            fromData: async (data, dataOffset = 0) => {
                const bufferOffset = bufferSegment.offset ?? 0
                const size = bufferSegment.size ?? Math.min(this.size - bufferOffset, data.byteLength - dataOffset)
                const absoluteDataOffset = data.byteOffset + dataOffset
                const validBufferOffset = lowerMultipleOf(4, bufferOffset)
                const offsetCorrection = bufferOffset - validBufferOffset
                const validDataOffset = absoluteDataOffset - offsetCorrection
                const validSize = upperMultipleOf(4, size + offsetCorrection)
                this.device.wrapped.queue.writeBuffer(this.wrapped, validBufferOffset, data.buffer, validDataOffset, validSize)
            }
        };
    }

    get(bufferSegment: DataSegment = {}): BufferReader {
        return {
            asData: (data, dataOffset = 0) => {
                const dataSize = data?.byteLength ?? this.size
                const bufferOffset = bufferSegment.offset ?? 0
                const size = bufferSegment.size ?? Math.min(this.size - bufferOffset, dataSize - dataOffset)
                const temp = this.device.readBuffer(size, `${this.label}-temp`);
                try {
                    this.copy(bufferSegment).to(temp);
                    return temp.get().asData(data, dataOffset);
                } finally {
                    temp.destroy();
                }
            }
        }
    }

    static instances<D extends DataBufferDescriptors>(device: Device, descriptors: D, labelPrefix?: string): DataBuffers<D> {
        const result: Partial<DataBuffers<D>> = {};
        for (const k in descriptors) {
            result[k] = DataBuffer.instance(device, withLabel(descriptors[k], labelPrefix, k));
        }
        return result as DataBuffers<D>;
    }


    static instance<D extends DataBufferDescriptor>(device: Device, descriptor: D): DataBuffer {
        return new DataBuffer(device, descriptor);
    }

}

export type BufferUsage = StrictExclude<(keyof GPUBufferUsage), "MAP_READ" | "MAP_WRITE" | "COPY_SRC" | "COPY_DST">[]
export type DataBuffers<D extends DataBufferDescriptors> = ReplaceValues<D, DataBuffer>;
export type DataBufferDescriptors = Record<string, DataBufferDescriptor>;
export type DataBufferDescriptor = {
    label?: string
    usage: BufferUsage;
} & ({
    data: DataView;
    size?: never;
} | {
    size: number;
    data?: never;
});

export class ReadBuffer extends AbstractBuffer implements DestinationBuffer {

    private mapper;

    private constructor(device: Device, size: number, label?: string) {
        super(device, newBlankBuffer(device, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, size, label));
        this.mapper = new BufferMapper(this.wrapped, GPUMapMode.READ);
    }

    copy(bufferSegment: DataSegment = {}): SourceClause<void> {
        return copy(bufferSegment.size).to(this, bufferSegment.offset);
    }

    get(bufferSegment: DataSegment = {}): BufferReader {
        return {
            asData: async (data?: DataView, dataOffset = 0) => {
                const bufferOffset = bufferSegment.offset ?? 0
                const dataView = data ?? new DataView(new ArrayBuffer(bufferSegment.size ?? this.size - bufferOffset))
                const size = bufferSegment.size ?? Math.min(this.size - bufferOffset, dataView.byteLength - dataOffset)
                const src = await this.mapper.mapAsync(bufferOffset, size);
                const dst = new Uint8Array(dataView.buffer, dataView.byteOffset + dataOffset, size);
                dst.set(src);
                return dataView;
            },
        };
    }

    static instances<D extends ReadBufferDescriptors>(device: Device, descriptors: D, labelPrefix?: string): ReadBuffers<D> {
        const result: Partial<ReadBuffers<D>> = {};
        for (const k in descriptors) {
            result[k] = ReadBuffer.instance(device, descriptors[k], label(labelPrefix, k));
        }
        return result as ReadBuffers<D>;
    }

    static instance(device: Device, size: number, label?: string): ReadBuffer {
        return new ReadBuffer(device, size, label);
    }

}

export type ReadBuffers<D extends ReadBufferDescriptors> = ReplaceValues<D, ReadBuffer>;
export type ReadBufferDescriptors = Record<string, number>;

export class WriteBuffer extends AbstractBuffer implements SourceBuffer {
    
    private mapper: BufferMapper;
    
    private constructor(device: Device, readonly data: DataView, label?: string) {
        super(device, newInitializedBuffer(device, GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, data, label));
        this.data = data;
        this.mapper = new BufferMapper(this.wrapped, GPUMapMode.WRITE);
    }

    copy(bufferSegment: DataSegment = {}): DestinationClause<void> {
        return copy(bufferSegment.size).from(this, bufferSegment.offset);
    }

    set(bufferSegment: DataSegment = {}): BufferWriter {
        return {
            fromData: async (data, dataOffset = 0) => {
                const bufferOffset = bufferSegment.offset ?? 0
                const size = bufferSegment.size ?? Math.min(this.size - bufferOffset, data.byteLength - dataOffset)
                const dst = await this.mapper.mapAsync(bufferOffset, size);
                const src = new Uint8Array(data.buffer, data.byteOffset + dataOffset, size);
                dst.set(src);
            }
        };
    }

    static instances<D extends WriteBufferDescriptors>(device: Device, descriptors: D, labelPrefix?: string): WriteBuffers<D> {
        const result: Partial<WriteBuffers<D>> = {};
        for (const k in descriptors) {
            result[k] = WriteBuffer.instance(device, descriptors[k], label(labelPrefix, k));
        }
        return result as WriteBuffers<D>;
    }

    static instance(device: Device, data: DataView, label?: string): WriteBuffer {
        return new WriteBuffer(device, data, label);
    }

}

export type WriteBuffers<D extends WriteBufferDescriptors> = ReplaceValues<D, WriteBuffer>;
export type WriteBufferDescriptors = Record<string, DataView>;

export function copy(segmentSize?: number): SourceClause<DestinationClause<void>> & DestinationClause<SourceClause<void>> {
    return {
        from: (srcBuffer, srcOffset = 0) => ({
            to: (dstBuffer, dstOffset = 0) => {
                const size = segmentSize ?? Math.min(srcBuffer.size - srcOffset, dstBuffer.size - dstOffset)
                const srcValidOffset = lowerMultipleOf(4, srcOffset)
                const dstValidOffset = lowerMultipleOf(4, dstOffset)
                const srcOffsetCorrection = srcOffset - srcValidOffset
                const dstOffsetCorrection = dstOffset - dstValidOffset
                if (srcOffsetCorrection !== dstOffsetCorrection) {
                    throw new Error("Copying between unaligned buffers is not possible!")
                }
                const validSize = upperMultipleOf(4, size + srcOffsetCorrection)
                srcBuffer.device.enqueueCommands(`copy-${srcBuffer.label}-to-${dstBuffer.label}`, encoder => {
                    encoder.encoder.copyBufferToBuffer(srcBuffer.wrapped, srcValidOffset, dstBuffer.wrapped, dstValidOffset, validSize)
                });
            }
        }),
        to: (dstBuffer, dstOffset = 0) => ({
            from: (srcBuffer, srcOffset = 0) => {
                copy(segmentSize).from(srcBuffer, srcOffset).to(dstBuffer, dstOffset)
            }
        })
    };
}

export type DestinationClause<T> = {
    to: (buffer: DestinationBuffer, offset?: number) => T;
};

export type SourceClause<T> = {
    from: (buffer: SourceBuffer, offset?: number) => T;
};

export type BufferReader = {
    asData: (data?: DataView, offset?: number) => Promise<DataView>;
};

export type BufferWriter = {
    fromData: (data: DataView, offset?: number) => Promise<void>;
};


class BufferMapper {

    private range: [number, number] = [this.buffer.size, 0]
    private promise = Promise.resolve();

    constructor(private buffer: GPUBuffer, private mode: GPUMapModeFlags) {
    }

    private request(offset: number, size: number, resolve: () => void, reject: (e: any) => void) {
        const mappingScheduled = this.range[1] > this.range[0];
        this.range = [Math.min(this.range[0], offset), Math.max(this.range[1], offset + size)];
        return (mappingScheduled ? this.promise : this.scheduleMapping()).then(resolve, e => {
            reject(e)
            return Promise.reject(e)
        });
    }

    private scheduleMapping() {
        const callback = () => new Promise<void>((resolve, reject) => {
            setTimeout(() => {
                this.buffer.mapAsync(this.mode, this.range[0], this.range[1] - this.range[0]).then(resolve, reject);
                this.range = [this.buffer.size, 0];
                this.promise = this.promise.finally(() => this.buffer.unmap());
            });
        });
        return this.promise = this.promise.then(callback, callback);
    }

    async mapAsync(offset: number, size: number) {
        const validOffset = lowerMultipleOf(8, offset);
        const offsetCorrection = offset - validOffset;
        const validSize = upperMultipleOf(4, size + offsetCorrection);
        return new Promise<Uint8Array>((resolve, reject) => this.request(validOffset, validSize, () => {
            const range = this.buffer.getMappedRange(validOffset, validSize)
            resolve(new Uint8Array(range.slice(), offsetCorrection, size))
        }, reject));
    }
}

function newBlankBuffer(device: Device, usage: GPUBufferUsageFlags, size: number, label?: string) {
    return device.wrapped.createBuffer({
        size: upperMultipleOf(4, size),
        usage,
        label
    });
}

function newInitializedBuffer(device: Device, usage: GPUBufferUsageFlags, data: DataView, label?: string) {
    const buffer = device.wrapped.createBuffer({
        mappedAtCreation: true,
        size: upperMultipleOf(4, data.byteLength),
        usage,
        label
    });
    const range = buffer.getMappedRange(0, buffer.size);
    const src = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const dst = new Uint8Array(range, 0, data.byteLength);
    dst.set(src);
    buffer.unmap();
    return buffer;
}

function usageFlags(descriptor: DataBufferDescriptor) {
    return GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | descriptor.usage
        .map(u => GPUBufferUsage[u])
        .reduce((u1, u2) => u1 | u2, 0);
}

function upperMultipleOf(n: number, value: number): number {
    return Math.ceil(value / n) * n
}

function lowerMultipleOf(n: number, value: number): number {
    return Math.floor(value / n) * n
}