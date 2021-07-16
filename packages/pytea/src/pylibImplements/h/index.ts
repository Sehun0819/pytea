import {
    HArgmaxr,
    HAveragepool,
    HAveragepoolp,
    HBop,
    HConcat,
    HConvol,
    HConvolp,
    HEBopType,
    HEUopType,
    HExpr,
    HMap,
    HMatmul,
    HMax,
    HMaxpool,
    HMaxpoolp,
    HMaxr,
    HNat,
    HReshape,
    HSumr,
    HTranspose,
    HVarTensor,
} from 'src/backend/hExpression';

import { fetchAddr } from '../../backend/backUtils';
import { Context, ContextSet } from '../../backend/context';
import { CodeSource, ShValue, SVObject, SVTorchGraph, SVType } from '../../backend/sharpValues';
import { ExpNum, ExpShape, NumOpType } from '../../backend/symExpressions';
import { LCImpl } from '..';
import { LCBase } from '../libcall';

export namespace HLCImpl {
    export function plain(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx
                .warnSizeWithMsg(
                    `from 'LibCall.h.plain': got insufficient number of argument: ${params.length}`,
                    source
                )
                .toSet();
        }

        const heap = ctx.heap;
        const argsAddr = params[0];

        // tensorInit is always used in Tensor.__init__ -> force casting
        const args = fetchAddr(argsAddr, heap)! as SVObject;

        // if first argument is object that has 'shape'
        const firstArg = fetchAddr(args.getIndice(0), heap);
        if (firstArg?.type === SVType.Object) {
            // if first argument is shaped value, cast it to Tensor
            let mayShaped: ShValue | undefined = firstArg;

            if (!mayShaped.shape) {
                mayShaped = fetchAddr(firstArg.getAttr('shape'), heap);
            }

            if (mayShaped?.type === SVType.Object && mayShaped?.shape !== undefined) {
                return SVTorchGraph.create(ctx, HVarTensor.createPlain(mayShaped.shape, source), source).toSet();
            }

            // else, check value is list of ... list of number
            const structure: (number | ExpNum)[] = [];
            let obj: ShValue | undefined = firstArg;
            let length = fetchAddr(firstArg.getAttr('$length'), heap);
            let shaped = true;

            if (length && length.type === SVType.Int) {
                // if argument is list of ... list of number, return that shape
                structure.push(length.value);
                obj = fetchAddr(obj.getIndice(0), heap);

                // simply fetch only first values
                while (obj?.type === SVType.Object) {
                    length = fetchAddr(obj.getAttr('$length'), heap);
                    if (length?.type === SVType.Int) {
                        structure.push(length.value);
                        obj = fetchAddr(obj.getIndice(0), heap);
                    } else {
                        shaped = false;
                        break;
                    }
                }

                // traversed list and ends with integer or float
                if (shaped && (obj?.type === SVType.Int || obj?.type === SVType.Float)) {
                    const shape = ExpShape.fromConst(structure.length, structure, source);
                    return SVTorchGraph.create(ctx, HVarTensor.createPlain(shape, source), source).toSet();
                }
            }
        }

        // if varargs is list of integer, parseSize
        return ctx.parseSize(argsAddr, source).map((ctx) => {
            let shape: ExpShape;
            let newCtx: Context<any> = ctx;
            if (typeof ctx.retVal === 'string') {
                newCtx = ctx.warnWithMsg(ctx.retVal, source).genIntGte('tempRank', 0, source);
                shape = ExpShape.fromSymbol(newCtx.genSymShape('tempShape', newCtx.retVal, source));
            } else {
                shape = ctx.retVal;
            }

            return SVTorchGraph.create(ctx, HVarTensor.createPlain(shape, source), source);
        });
    }

    export function add(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.add': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr] = params;
        const leftObj = fetchAddr(leftAddr, heap);
        const rightObj = fetchAddr(rightAddr, heap);

        if (leftObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.add': got invalid tensor: ${leftAddr}`, source).toSet();
        } else if (
            rightObj?.type !== SVType.TorchGraph &&
            rightObj?.type !== SVType.Int &&
            rightObj?.type !== SVType.Float
        ) {
            return ctx.failWithMsg(`from 'LibCall.h.add': got invalid tensor: ${rightAddr}`, source).toSet();
        }

        const left = leftObj.hexpr;
        let right: HExpr | undefined;
        if (rightObj.type === SVType.TorchGraph) {
            right = rightObj.hexpr;
        } else {
            const num = rightObj.value;
            if (typeof num !== 'number' && num.opType !== NumOpType.Const) {
                return ctx.failWithMsg(`from 'LibCall.h.add': got invalid number: ${rightAddr}`, source).toSet();
            }
            right = HNat.create(typeof num === 'number' ? num : num.value, source);
        }

        if (!left || !right) {
            return ctx.failWithMsg(`from 'LibCall.h.add': got invalid argument`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HBop.create(HEBopType.Add, left, right, source), source).toSet();
    }

    export function sub(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.sub': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr] = params;
        const leftObj = fetchAddr(leftAddr, heap);
        const rightObj = fetchAddr(rightAddr, heap);

        if (leftObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.sub': got invalid tensor: ${leftAddr}`, source).toSet();
        } else if (
            rightObj?.type !== SVType.TorchGraph &&
            rightObj?.type !== SVType.Int &&
            rightObj?.type !== SVType.Float
        ) {
            return ctx.failWithMsg(`from 'LibCall.h.sub': got invalid tensor: ${rightAddr}`, source).toSet();
        }

        const left = leftObj.hexpr;
        let right: HExpr | undefined;
        if (rightObj.type === SVType.TorchGraph) {
            right = rightObj.hexpr;
        } else {
            const num = rightObj.value;
            if (typeof num !== 'number' && num.opType !== NumOpType.Const) {
                return ctx.failWithMsg(`from 'LibCall.h.sub': got invalid number: ${rightAddr}`, source).toSet();
            }
            right = HNat.create(typeof num === 'number' ? num : num.value, source);
        }

        if (!left || !right) {
            return ctx.failWithMsg(`from 'LibCall.h.sub': got invalid argument`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HBop.create(HEBopType.Sub, left, right, source), source).toSet();
    }

    export function mul(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.mul': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr] = params;
        const leftObj = fetchAddr(leftAddr, heap);
        const rightObj = fetchAddr(rightAddr, heap);

        if (leftObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.mul': got invalid tensor: ${leftAddr}`, source).toSet();
        } else if (
            rightObj?.type !== SVType.TorchGraph &&
            rightObj?.type !== SVType.Int &&
            rightObj?.type !== SVType.Float
        ) {
            return ctx.failWithMsg(`from 'LibCall.h.mul': got invalid tensor: ${rightAddr}`, source).toSet();
        }

        const left = leftObj.hexpr;
        let right: HExpr | undefined;
        if (rightObj.type === SVType.TorchGraph) {
            right = rightObj.hexpr;
        } else {
            const num = rightObj.value;
            if (typeof num !== 'number' && num.opType !== NumOpType.Const) {
                return ctx.failWithMsg(`from 'LibCall.h.mul': got invalid number: ${rightAddr}`, source).toSet();
            }
            right = HNat.create(typeof num === 'number' ? num : num.value, source);
        }

        if (!left || !right) {
            return ctx.failWithMsg(`from 'LibCall.h.mul': got invalid argument`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HBop.create(HEBopType.Mul, left, right, source), source).toSet();
    }

    export function maxt(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.max': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr] = params;
        const leftObj = fetchAddr(leftAddr, heap);
        const rightObj = fetchAddr(rightAddr, heap);

        if (leftObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.max': got invalid tensor: ${leftAddr}`, source).toSet();
        } else if (rightObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.max': got invalid tensor: ${rightAddr}`, source).toSet();
        }

        const left = leftObj.hexpr;
        const right = rightObj.hexpr;

        if (!left || !right) {
            return ctx.failWithMsg(`from 'LibCall.h.max': got invalid tensor`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HBop.create(HEBopType.Max, left, right, source), source).toSet();
    }

    export function matmul(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.matmul': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, rightAddr] = params;
        const leftObj = fetchAddr(leftAddr, heap);
        const rightObj = fetchAddr(rightAddr, heap);

        if (leftObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.matmul': got invalid tensor: ${leftAddr}`, source).toSet();
        } else if (rightObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.matmul': got invalid tensor: ${rightAddr}`, source).toSet();
        }

        const left = leftObj.hexpr;
        const right = rightObj.hexpr;

        if (!left || !right) {
            return ctx.failWithMsg(`from 'LibCall.h.matmul': got invalid tensor`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HMatmul.create(left, right, source), source).toSet();
    }

    export function concat(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.concat': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [leftAddr, axisAddr, rightAddr] = params;
        const leftObj = fetchAddr(leftAddr, heap);
        const axisVal = fetchAddr(axisAddr, heap);
        const rightObj = fetchAddr(rightAddr, heap);

        if (leftObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.concat': got invalid tensor: ${leftAddr}`, source).toSet();
        } else if (rightObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.concat': got invalid tensor: ${rightAddr}`, source).toSet();
        } else if (axisVal?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.h.concat': got invalid axis: ${axisAddr}`, source).toSet();
        }

        const left = leftObj.hexpr;
        const right = rightObj.hexpr;

        if (!left || !right) {
            return ctx.failWithMsg(`from 'LibCall.h.sub': got invalid tensor`, source).toSet();
        } else if (typeof axisVal.value !== 'number' && axisVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.sub': got invalid axis`, source).toSet();
        }

        const axis = typeof axisVal.value === 'number' ? axisVal.value : axisVal.value.value;

        return SVTorchGraph.create(ctx, HConcat.create(left, axis, right, source), source).toSet();
    }

    export function reshape(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.reshape': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [baseAddr, shapeAddr] = params;

        const baseObj = fetchAddr(baseAddr, heap);
        if (baseObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.reshape': ${baseAddr}`, source).toSet();
        }

        const base = baseObj.hexpr;
        if (!base) {
            return ctx.failWithMsg(`from 'LibCall.h.reshape': got invalid tensor`, source).toSet();
        }

        // if first argument is object that has 'shape'
        const shapeObj = fetchAddr(shapeAddr, heap);
        if (shapeObj?.type !== SVType.Object) {
            return ctx.failWithMsg(`from 'LibCall.h.reshape': got invalid shape`, source).toSet();
        }
        // if first argument is shaped value, cast it to Tensor
        let mayShaped: ShValue | undefined = shapeObj;

        if (!mayShaped.shape) {
            mayShaped = fetchAddr(shapeObj.getAttr('shape'), heap);
        }

        if (mayShaped?.type !== SVType.Object || mayShaped?.shape === undefined) {
            return ctx.failWithMsg(`from 'LibCall.h.reshape': got invalid shape`, source).toSet();
        }
        return SVTorchGraph.create(ctx, HReshape.create(base, mayShaped.shape, source), source).toSet();
    }

    export function transpose(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.transpose': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [baseAddr, dim0Addr, dim1Addr] = params;
        const baseObj = fetchAddr(baseAddr, heap);
        const dim0Val = fetchAddr(dim0Addr, heap);
        const dim1Val = fetchAddr(dim1Addr, heap);

        if (baseObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.transpose': got invalid tensor: ${baseAddr}`, source).toSet();
        } else if (dim0Val?.type !== SVType.Int) {
            return ctx
                .failWithMsg(`from 'LibCall.h.transpose': got invalid kernel height: ${dim0Addr}`, source)
                .toSet();
        } else if (dim1Val?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.h.transpose': got invalid kernel width: ${dim1Addr}`, source).toSet();
        }

        const base = baseObj.hexpr;

        if (!base) {
            return ctx.failWithMsg(`from 'LibCall.h.transpose': got invalid tensor`, source).toSet();
        } else if (typeof dim0Val.value !== 'number' && dim0Val.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.transpose': got invalid axis`, source).toSet();
        } else if (typeof dim1Val.value !== 'number' && dim1Val.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.transpose': got invalid axis`, source).toSet();
        }

        const dim0 = typeof dim0Val.value === 'number' ? dim0Val.value : dim0Val.value.value;
        const dim1 = typeof dim1Val.value === 'number' ? dim1Val.value : dim1Val.value.value;

        return SVTorchGraph.create(ctx, HTranspose.create(base, dim0, dim1, source), source).toSet();
    }

    export function max(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.max': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [baseAddr] = params;
        const baseObj = fetchAddr(baseAddr, heap);

        if (baseObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.max': got invalid tensor: ${baseAddr}`, source).toSet();
        }

        const base = baseObj.hexpr;

        if (!base) {
            return ctx.failWithMsg(`from 'LibCall.h.max': got invalid tensor`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HMax.create(base, source), source).toSet();
    }

    export function maxr(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.maxr': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [baseAddr, axisAddr] = params;
        const baseObj = fetchAddr(baseAddr, heap);
        const axisVal = fetchAddr(axisAddr, heap);

        if (baseObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.maxr': got invalid tensor: ${baseAddr}`, source).toSet();
        } else if (axisVal?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.h.maxr': got invalid axis: ${axisAddr}`, source).toSet();
        }

        const base = baseObj.hexpr;

        if (!base) {
            return ctx.failWithMsg(`from 'LibCall.h.maxr': got invalid tensor`, source).toSet();
        } else if (typeof axisVal.value !== 'number' && axisVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.maxr': got invalid axis`, source).toSet();
        }

        const axis = typeof axisVal.value === 'number' ? axisVal.value : axisVal.value.value;

        return SVTorchGraph.create(ctx, HMaxr.create(base, axis, source), source).toSet();
    }

    export function argmaxr(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.argmaxr': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [baseAddr, axisAddr] = params;
        const baseObj = fetchAddr(baseAddr, heap);
        const axisVal = fetchAddr(axisAddr, heap);

        if (baseObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.argmaxr': got invalid tensor: ${baseAddr}`, source).toSet();
        } else if (axisVal?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.h.argmaxr': got invalid axis: ${axisAddr}`, source).toSet();
        }

        const base = baseObj.hexpr;

        if (!base) {
            return ctx.failWithMsg(`from 'LibCall.h.argmaxr': got invalid tensor`, source).toSet();
        } else if (typeof axisVal.value !== 'number' && axisVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.argmaxr': got invalid axis`, source).toSet();
        }

        const axis = typeof axisVal.value === 'number' ? axisVal.value : axisVal.value.value;

        return SVTorchGraph.create(ctx, HArgmaxr.create(base, axis, source), source).toSet();
    }

    export function sumr(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.sumr': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [baseAddr, axisAddr] = params;
        const baseObj = fetchAddr(baseAddr, heap);
        const axisVal = fetchAddr(axisAddr, heap);

        if (baseObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.sumr': got invalid tensor: ${baseAddr}`, source).toSet();
        } else if (axisVal?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.h.sumr': got invalid axis: ${axisAddr}`, source).toSet();
        }

        const base = baseObj.hexpr;

        if (!base) {
            return ctx.failWithMsg(`from 'LibCall.h.sumr': got invalid tensor`, source).toSet();
        } else if (typeof axisVal.value !== 'number' && axisVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.sumr': got invalid axis`, source).toSet();
        }

        const axis = typeof axisVal.value === 'number' ? axisVal.value : axisVal.value.value;

        return SVTorchGraph.create(ctx, HSumr.create(base, axis, source), source).toSet();
    }

    export function convol(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.convol': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [fmapAddr, kernelAddr] = params;
        const fmapObj = fetchAddr(fmapAddr, heap);
        const kernelObj = fetchAddr(kernelAddr, heap);

        if (fmapObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.convol': got invalid tensor: ${fmapAddr}`, source).toSet();
        } else if (kernelObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.convol': got invalid tensor: ${kernelAddr}`, source).toSet();
        }

        const fmap = fmapObj.hexpr;
        const kernel = kernelObj.hexpr;

        if (!fmap || !kernel) {
            return ctx.failWithMsg(`from 'LibCall.h.convol': got invalid tensor`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HConvol.create(fmap, kernel, source), source).toSet();
    }

    export function convolp(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.convolp': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [fmapAddr, kernelAddr] = params;
        const fmapObj = fetchAddr(fmapAddr, heap);
        const kernelObj = fetchAddr(kernelAddr, heap);

        if (fmapObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.convolp': got invalid tensor: ${fmapAddr}`, source).toSet();
        } else if (kernelObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.convolp': got invalid tensor: ${kernelAddr}`, source).toSet();
        }

        const fmap = fmapObj.hexpr;
        const kernel = kernelObj.hexpr;

        if (!fmap || !kernel) {
            return ctx.failWithMsg(`from 'LibCall.h.convolp': got invalid tensor`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HConvolp.create(fmap, kernel, source), source).toSet();
    }

    export function maxpool(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.maxpool': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [fmapAddr, khAddr, kwAddr] = params;
        const fmapObj = fetchAddr(fmapAddr, heap);
        const khVal = fetchAddr(khAddr, heap);
        const kwVal = fetchAddr(kwAddr, heap);

        if (fmapObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpool': got invalid tensor: ${fmapAddr}`, source).toSet();
        } else if (khVal?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpool': got invalid kernel height: ${khAddr}`, source).toSet();
        } else if (kwVal?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpool': got invalid kernel width: ${kwAddr}`, source).toSet();
        }

        const fmap = fmapObj.hexpr;

        if (!fmap) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpool': got invalid tensor`, source).toSet();
        } else if (typeof khVal.value !== 'number' && khVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpool': got invalid axis`, source).toSet();
        } else if (typeof kwVal.value !== 'number' && kwVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpool': got invalid axis`, source).toSet();
        }

        const kh = typeof khVal.value === 'number' ? khVal.value : khVal.value.value;
        const kw = typeof kwVal.value === 'number' ? kwVal.value : kwVal.value.value;

        return SVTorchGraph.create(ctx, HMaxpool.create(fmap, kh, kw, source), source).toSet();
    }

    export function maxpoolp(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.maxpoolp': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [fmapAddr, khAddr, kwAddr] = params;
        const fmapObj = fetchAddr(fmapAddr, heap);
        const khVal = fetchAddr(khAddr, heap);
        const kwVal = fetchAddr(kwAddr, heap);

        if (fmapObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpoolp': got invalid tensor: ${fmapAddr}`, source).toSet();
        } else if (khVal?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpoolp': got invalid kernel height: ${khAddr}`, source).toSet();
        } else if (kwVal?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpoolp': got invalid kernel width: ${kwAddr}`, source).toSet();
        }

        const fmap = fmapObj.hexpr;

        if (!fmap) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpoolp': got invalid tensor`, source).toSet();
        } else if (typeof khVal.value !== 'number' && khVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpoolp': got invalid axis`, source).toSet();
        } else if (typeof kwVal.value !== 'number' && kwVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.maxpoolp': got invalid axis`, source).toSet();
        }

        const kh = typeof khVal.value === 'number' ? khVal.value : khVal.value.value;
        const kw = typeof kwVal.value === 'number' ? kwVal.value : kwVal.value.value;

        return SVTorchGraph.create(ctx, HMaxpoolp.create(fmap, kh, kw, source), source).toSet();
    }

    export function averagepool(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.averagepool': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [fmapAddr, khAddr, kwAddr] = params;
        const fmapObj = fetchAddr(fmapAddr, heap);
        const khVal = fetchAddr(khAddr, heap);
        const kwVal = fetchAddr(kwAddr, heap);

        if (fmapObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.averagepool': got invalid tensor: ${fmapAddr}`, source).toSet();
        } else if (khVal?.type !== SVType.Int) {
            return ctx
                .failWithMsg(`from 'LibCall.h.averagepool': got invalid kernel height: ${khAddr}`, source)
                .toSet();
        } else if (kwVal?.type !== SVType.Int) {
            return ctx.failWithMsg(`from 'LibCall.h.averagepool': got invalid kernel width: ${kwAddr}`, source).toSet();
        }

        const fmap = fmapObj.hexpr;

        if (!fmap) {
            return ctx.failWithMsg(`from 'LibCall.h.averagepool': got invalid tensor`, source).toSet();
        } else if (typeof khVal.value !== 'number' && khVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.averagepool': got invalid axis`, source).toSet();
        } else if (typeof kwVal.value !== 'number' && kwVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.averagepool': got invalid axis`, source).toSet();
        }

        const kh = typeof khVal.value === 'number' ? khVal.value : khVal.value.value;
        const kw = typeof kwVal.value === 'number' ? kwVal.value : kwVal.value.value;

        return SVTorchGraph.create(ctx, HAveragepool.create(fmap, kh, kw, source), source).toSet();
    }

    export function averagepoolp(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.averagepoolp': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [fmapAddr, khAddr, kwAddr] = params;
        const fmapObj = fetchAddr(fmapAddr, heap);
        const khVal = fetchAddr(khAddr, heap);
        const kwVal = fetchAddr(kwAddr, heap);

        if (fmapObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.averagepoolp': got invalid tensor: ${fmapAddr}`, source).toSet();
        } else if (khVal?.type !== SVType.Int) {
            return ctx
                .failWithMsg(`from 'LibCall.h.averagepoolp': got invalid kernel height: ${khAddr}`, source)
                .toSet();
        } else if (kwVal?.type !== SVType.Int) {
            return ctx
                .failWithMsg(`from 'LibCall.h.averagepoolp': got invalid kernel width: ${kwAddr}`, source)
                .toSet();
        }

        const fmap = fmapObj.hexpr;

        if (!fmap) {
            return ctx.failWithMsg(`from 'LibCall.h.averagepoolp': got invalid tensor`, source).toSet();
        } else if (typeof khVal.value !== 'number' && khVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.averagepoolp': got invalid axis`, source).toSet();
        } else if (typeof kwVal.value !== 'number' && kwVal.value.opType !== NumOpType.Const) {
            return ctx.failWithMsg(`from 'LibCall.h.averagepoolp': got invalid axis`, source).toSet();
        }

        const kh = typeof khVal.value === 'number' ? khVal.value : khVal.value.value;
        const kw = typeof kwVal.value === 'number' ? kwVal.value : kwVal.value.value;

        return SVTorchGraph.create(ctx, HAveragepoolp.create(fmap, kh, kw, source), source).toSet();
    }

    export function relu(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.relu': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [baseAddr] = params;
        const baseObj = fetchAddr(baseAddr, heap);

        if (baseObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.relu': got invalid tensor: ${baseAddr}`, source).toSet();
        }

        const base = baseObj.hexpr;

        if (!base) {
            return ctx.failWithMsg(`from 'LibCall.h.relu': got invalid tensor`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HMap.create(HEUopType.Relu, base, source), source).toSet();
    }

    export function sigmoid(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.sigmoid': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [baseAddr] = params;
        const baseObj = fetchAddr(baseAddr, heap);

        if (baseObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.sigmoid': got invalid tensor: ${baseAddr}`, source).toSet();
        }

        const base = baseObj.hexpr;

        if (!base) {
            return ctx.failWithMsg(`from 'LibCall.h.sigmoid': got invalid tensor`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HMap.create(HEUopType.Sigmoid, base, source), source).toSet();
    }

    export function rtangent(ctx: Context<LCBase.ExplicitParams>, source: CodeSource | undefined): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 1) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.h.rtangent': got insufficient number of argument: ${params.length}`,
                source
            );
        }

        const heap = ctx.heap;
        const [baseAddr] = params;
        const baseObj = fetchAddr(baseAddr, heap);

        if (baseObj?.type !== SVType.TorchGraph) {
            return ctx.failWithMsg(`from 'LibCall.h.rtangent': got invalid tensor: ${baseAddr}`, source).toSet();
        }

        const base = baseObj.hexpr;

        if (!base) {
            return ctx.failWithMsg(`from 'LibCall.h.rtangent': got invalid tensor`, source).toSet();
        }

        return SVTorchGraph.create(ctx, HMap.create(HEUopType.Rtangent, base, source), source).toSet();
    }

    export const libCallImpls: { [key: string]: LCImpl } = {
        plain,
        matmul,
        add,
        sub,
        mul,
        maxt,
        reshape,
        transpose,
        max,
        maxr,
        argmaxr,
        sumr,
        convol,
        convolp,
        maxpool,
        maxpoolp,
        averagepool,
        averagepoolp,
        relu,
        sigmoid,
        rtangent,
    };
}

export const libCallMap: Map<string, LCImpl> = new Map([...Object.entries(HLCImpl.libCallImpls)]);
