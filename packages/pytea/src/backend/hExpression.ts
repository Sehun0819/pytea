/*
 * torchStatements.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Expressions and statements for PyTea internal languages.
 */
import { Map } from 'immutable';

import { CodeSource } from './sharpValues';
import { ExpShape } from './symExpressions';

export type HExpr = HNat | HVar | HVarTensor | HConstTensor | HLet | HTop;

export const enum HEType {
    Nat,
    Var,
    VarTensor,
    ConstTensor,
    Let,
    Top,
}

export const enum HVarTensorType {
    Plain,
    Cipher,
}

export const enum HETopType {
    Bop,
    Matmul,
    Concat,
    Reshape,
    Transpose,
    Max,
    Maxr,
    Argmaxr,
    Sumr,
    Convol,
    Convolp,
    Maxpool,
    Maxpoolp,
    Averagepool,
    Averagepoolp,
    Map,
}

// element-wise Tensor bop
export const enum HEBopType {
    Add,
    Sub,
    Mul,
    Max,
}

// element-wise Tensor uop
export const enum HEUopType {
    Relu,
    Sigmoid,
    Rtangent,
}

interface HExprBase {
    etype: HEType;
    source?: CodeSource;

    toString(): string;
}

const _indent = '    ';

/*
function shapeStr(shape: number[], indent = 0): string {
    const numStr = shape.map((num) => `${num}`).join(' ');
    return `${_indent.repeat(indent)}[${numStr}]`;
}*/

export namespace HExpr {
    export function toString(hexpr: HExpr): string {
        return toStrs(hexpr).join('\n');
    }

    export function toStrs(hexpr: HExpr): string[] {
        switch (hexpr.etype) {
            case HEType.Nat:
                return HNat.toStrs(hexpr);
            case HEType.Var:
                return HVar.toStrs(hexpr);
            case HEType.VarTensor:
                return HVarTensor.toStrs(hexpr);
            case HEType.ConstTensor:
                return HConstTensor.toStrs(hexpr);
            case HEType.Let:
                return HLet.toStrs(hexpr);
            case HEType.Top:
                return HTop.toStrs(hexpr);
        }
    }

    export function prefixStr(operator: string, operands: string[]): string[] {
        const operands_ = operands.map((str) => `${_indent}${str}`);
        return [operator].concat(operands_);
    }

    export function infixStr(op: string, left: string[], right: string[]): string[] {
        return left.concat([op]).concat(right);
    }

    export function nameResolve(hexpr: HExpr, name2path: Map<string, string>) {
        switch (hexpr.etype) {
            case HEType.VarTensor: {
                const pathStr = name2path.get(hexpr.tname);
                if (pathStr === undefined) {
                    HVarTensor.toCipher(hexpr);
                } else {
                    HVarTensor.setName(hexpr, pathStr);
                }
                break;
            }
            case HEType.Let: {
                nameResolve(hexpr.scope, name2path);
                break;
            }
            case HEType.Top: {
                HTop.nameResolve(hexpr, name2path);
                break;
            }
            default:
                break;
        }
    }
}

export interface HNat extends HExprBase {
    etype: HEType.Nat;
    value: number;
}

export namespace HNat {
    export function create(value: number, source?: CodeSource): HNat {
        return {
            etype: HEType.Nat,
            value,
            source,
        };
    }

    export function toStrs(hexpr: HNat): string[] {
        return [`${hexpr.value}`];
    }
}

export interface HVar extends HExprBase {
    etype: HEType.Var;
    ident: string;
}

export namespace HVar {
    export function create(ident: string, source?: CodeSource): HVar {
        return {
            etype: HEType.Var,
            ident,
            source,
        };
    }

    export function toStrs(hexpr: HVar): string[] {
        return [`${hexpr.ident}`];
    }
}

let _cipherId = 0;
export function getNextCipherId(): number {
    return _cipherId++;
}

let _plainId = 0;
export function getNextPlainId(): number {
    return _plainId++;
}

export interface HVarTensor extends HExprBase {
    etype: HEType.VarTensor;
    ttype: HVarTensorType;
    tname: string;
    shape: ExpShape;
}

export namespace HVarTensor {
    export function createPlain(shape: ExpShape, source?: CodeSource): HVarTensor {
        return {
            etype: HEType.VarTensor,
            ttype: HVarTensorType.Plain,
            tname: `tempPlain${getNextPlainId()}`,
            shape,
            source,
        };
    }

    export function createCipher(shape: ExpShape, source?: CodeSource): HVarTensor {
        return {
            etype: HEType.VarTensor,
            ttype: HVarTensorType.Cipher,
            tname: `tempCipher${getNextCipherId()}`,
            shape,
            source,
        };
    }

    export function setName(tensor: HVarTensor, name: string) {
        tensor.tname = name;
    }

    export function toCipher(tensor: HVarTensor) {
        tensor.ttype = HVarTensorType.Cipher;
        tensor.tname = `tempCipher${getNextCipherId()}`;
    }

    export function toStrs(tensor: HVarTensor): string[] {
        const ttype = tensor.ttype === HVarTensorType.Plain ? 'plain' : 'cipher';
        const shape = ExpShape.toString(tensor.shape);
        return [`${ttype} ${tensor.tname}:${shape}`];
    }
}

// Not implemeted yet
export interface HConstTensor extends HExprBase {
    etype: HEType.ConstTensor;
    tensor: undefined;
    shape: number[];
}

export namespace HConstTensor {
    export function create(shape: number[], source?: CodeSource): HConstTensor {
        return {
            etype: HEType.ConstTensor,
            tensor: undefined,
            shape,
            source,
        };
    }

    export function toStrs(hexpr: HConstTensor): string[] {
        return [`TODO: HConstTensor`];
    }
}

export interface HLet extends HExprBase {
    etype: HEType.Let;
    vardecs: [string, HExpr][];
    scope: HExpr;
}

export namespace HLet {
    export function create(vardecs: [string, HExpr][], scope: HExpr, source?: CodeSource): HLet {
        return {
            etype: HEType.Let,
            vardecs,
            scope,
            source,
        };
    }

    export function toStrs(hexpr: HLet): string[] {
        return [`TODO: HConstTensor`];
    }
}

export interface HTop extends HExprBase {
    etype: HEType.Top;
    topType: HETopType;
}

export namespace HTop {
    export function bop(bopType: HEBopType, left: HExpr, right: HExpr, source?: CodeSource): HBop {
        return {
            etype: HEType.Top,
            topType: HETopType.Bop,
            bopType,
            left,
            right,
            source,
        };
    }

    export function matmul(left: HExpr, right: HExpr, source?: CodeSource): HMatmul {
        return {
            etype: HEType.Top,
            topType: HETopType.Matmul,
            left,
            right,
            source,
        };
    }

    export function toStrs(htop: HTop): string[] {
        switch (htop.topType) {
            case HETopType.Bop:
                return HBop.toStrs(htop as HBop);
            case HETopType.Matmul:
                return HMatmul.toStrs(htop as HMatmul);
            case HETopType.Concat:
                return HConcat.toStrs(htop as HConcat);
            case HETopType.Reshape:
                return HReshape.toStrs(htop as HReshape);
            case HETopType.Transpose:
                return HTranspose.toStrs(htop as HTranspose);
            case HETopType.Max:
                return HMax.toStrs(htop as HMax);
            case HETopType.Maxr:
                return HMaxr.toStrs(htop as HMaxr);
            case HETopType.Argmaxr:
                return HArgmaxr.toStrs(htop as HArgmaxr);
            case HETopType.Sumr:
                return HSumr.toStrs(htop as HSumr);
            case HETopType.Convol:
                return HConvol.toStrs(htop as HConvol);
            case HETopType.Convolp:
                return HConvolp.toStrs(htop as HConvolp);
            case HETopType.Maxpool:
                return HMaxpool.toStrs(htop as HMaxpool);
            case HETopType.Maxpoolp:
                return HMaxpoolp.toStrs(htop as HMaxpoolp);
            case HETopType.Averagepool:
                return HAveragepool.toStrs(htop as HAveragepool);
            case HETopType.Averagepoolp:
                return HAveragepoolp.toStrs(htop as HAveragepoolp);
            case HETopType.Map:
                return HMap.toStrs(htop as HMap);
            default:
                // Can't reach here
                return [''];
        }
    }

    export function nameResolve(htop: HTop, name2path: Map<string, string>) {
        switch (htop.topType) {
            case HETopType.Bop: {
                HExpr.nameResolve((htop as HBop).left, name2path);
                HExpr.nameResolve((htop as HBop).right, name2path);
                break;
            }
            case HETopType.Matmul: {
                HExpr.nameResolve((htop as HMatmul).left, name2path);
                HExpr.nameResolve((htop as HMatmul).right, name2path);
                break;
            }
            case HETopType.Concat: {
                HExpr.nameResolve((htop as HConcat).left, name2path);
                HExpr.nameResolve((htop as HConcat).right, name2path);
                break;
            }
            case HETopType.Reshape: {
                HExpr.nameResolve((htop as HReshape).base, name2path);
                break;
            }
            case HETopType.Transpose: {
                HExpr.nameResolve((htop as HTranspose).base, name2path);
                break;
            }
            case HETopType.Max: {
                HExpr.nameResolve((htop as HMax).base, name2path);
                break;
            }
            case HETopType.Maxr: {
                HExpr.nameResolve((htop as HMaxr).base, name2path);
                break;
            }
            case HETopType.Argmaxr: {
                HExpr.nameResolve((htop as HArgmaxr).base, name2path);
                break;
            }
            case HETopType.Sumr: {
                HExpr.nameResolve((htop as HSumr).base, name2path);
                break;
            }
            case HETopType.Convol: {
                HExpr.nameResolve((htop as HConvol).fmap, name2path);
                HExpr.nameResolve((htop as HConvol).kernel, name2path);
                break;
            }
            case HETopType.Convolp: {
                HExpr.nameResolve((htop as HConvolp).fmap, name2path);
                HExpr.nameResolve((htop as HConvolp).kernel, name2path);
                break;
            }
            case HETopType.Maxpool: {
                HExpr.nameResolve((htop as HMaxpool).fmap, name2path);
                break;
            }
            case HETopType.Maxpoolp: {
                HExpr.nameResolve((htop as HMaxpoolp).fmap, name2path);
                break;
            }
            case HETopType.Averagepool: {
                HExpr.nameResolve((htop as HAveragepool).fmap, name2path);
                break;
            }
            case HETopType.Averagepoolp: {
                HExpr.nameResolve((htop as HAveragepoolp).fmap, name2path);
                break;
            }
            case HETopType.Map: {
                HExpr.nameResolve((htop as HMap).base, name2path);
                break;
            }
            default:
                // Can't reach here
                return '';
        }
    }
}

export interface HBop extends HTop {
    etype: HEType.Top;
    topType: HETopType.Bop;
    bopType: HEBopType;
    left: HExpr;
    right: HExpr;
}

export namespace HBop {
    export function create(bopType: HEBopType, left: HExpr, right: HExpr, source?: CodeSource): HBop {
        return {
            etype: HEType.Top,
            topType: HETopType.Bop,
            bopType,
            left,
            right,
            source,
        };
    }

    function bopChar(hbop: HBop): string {
        switch (hbop.bopType) {
            case HEBopType.Add:
                return '+';
            case HEBopType.Sub:
                return '-';
            case HEBopType.Mul:
                return '*';
            case HEBopType.Max:
                return 'maxt';
        }
    }

    export function toStrs(hbop: HBop): string[] {
        return HExpr.infixStr(bopChar(hbop), HExpr.toStrs(hbop.left), HExpr.toStrs(hbop.right));
    }
}

export interface HMatmul extends HTop {
    etype: HEType.Top;
    topType: HETopType.Matmul;
    left: HExpr;
    right: HExpr;
}

export namespace HMatmul {
    export function create(left: HExpr, right: HExpr, source?: CodeSource): HMatmul {
        return {
            etype: HEType.Top,
            topType: HETopType.Matmul,
            left,
            right,
            source,
        };
    }

    export function toStrs(hexpr: HMatmul): string[] {
        return HExpr.prefixStr('matmul', HExpr.toStrs(hexpr.left).concat(HExpr.toStrs(hexpr.right)));
    }
}

export interface HConcat extends HTop {
    etype: HEType.Top;
    topType: HETopType.Concat;
    left: HExpr;
    axis: number;
    right: HExpr;
}

export interface HConcat extends HTop {
    etype: HEType.Top;
    topType: HETopType.Concat;
    left: HExpr;
    axis: number;
    right: HExpr;
}

export namespace HConcat {
    export function create(left: HExpr, axis: number, right: HExpr, source?: CodeSource): HConcat {
        return {
            etype: HEType.Top,
            topType: HETopType.Concat,
            left,
            axis,
            right,
            source,
        };
    }

    export function toStrs(hexpr: HConcat): string[] {
        return HExpr.prefixStr(
            'concat',
            HExpr.toStrs(hexpr.left)
                .concat([`${hexpr.axis}`])
                .concat(HExpr.toStrs(hexpr.right))
        );
    }
}

export interface HReshape extends HTop {
    etype: HEType.Top;
    topType: HETopType.Reshape;
    base: HExpr;
    shape: ExpShape;
}

export namespace HReshape {
    export function create(base: HExpr, shape: ExpShape, source?: CodeSource): HReshape {
        return {
            etype: HEType.Top,
            topType: HETopType.Reshape,
            base,
            shape,
            source,
        };
    }

    export function toStrs(hexpr: HReshape): string[] {
        return HExpr.prefixStr('reshape', HExpr.toStrs(hexpr.base).concat([ExpShape.toString(hexpr.shape)]));
    }
}

export interface HTranspose extends HTop {
    etype: HEType.Top;
    topType: HETopType.Transpose;
    base: HExpr;
    dim0: number;
    dim1: number;
}

export namespace HTranspose {
    export function create(base: HExpr, dim0: number, dim1: number, source?: CodeSource): HTranspose {
        return {
            etype: HEType.Top,
            topType: HETopType.Transpose,
            base,
            dim0,
            dim1,
            source,
        };
    }

    export function toStrs(hexpr: HTranspose): string[] {
        return HExpr.prefixStr(
            'transpose',
            HExpr.toStrs(hexpr.base)
                .concat([`${hexpr.dim0}`])
                .concat([`${hexpr.dim1}`])
        );
    }
}

export interface HMax extends HTop {
    etype: HEType.Top;
    topType: HETopType.Max;
    base: HExpr;
}

export namespace HMax {
    export function create(base: HExpr, source?: CodeSource): HMax {
        return {
            etype: HEType.Top,
            topType: HETopType.Max,
            base,
            source,
        };
    }

    export function toStrs(hexpr: HMax): string[] {
        return HExpr.prefixStr('max', HExpr.toStrs(hexpr.base));
    }
}

export interface HMaxr extends HTop {
    etype: HEType.Top;
    topType: HETopType.Maxr;
    base: HExpr;
    axis: number;
}

export namespace HMaxr {
    export function create(base: HExpr, axis: number, source?: CodeSource): HMaxr {
        return {
            etype: HEType.Top,
            topType: HETopType.Maxr,
            base,
            axis,
            source,
        };
    }

    export function toStrs(hexpr: HMaxr): string[] {
        return HExpr.prefixStr('maxr', HExpr.toStrs(hexpr.base).concat([`${hexpr.axis}`]));
    }
}

export interface HArgmaxr extends HTop {
    etype: HEType.Top;
    topType: HETopType.Argmaxr;
    base: HExpr;
    axis: number;
}

export namespace HArgmaxr {
    export function create(base: HExpr, axis: number, source?: CodeSource): HArgmaxr {
        return {
            etype: HEType.Top,
            topType: HETopType.Argmaxr,
            base,
            axis,
            source,
        };
    }

    export function toStrs(hexpr: HArgmaxr): string[] {
        return HExpr.prefixStr('argmaxr', HExpr.toStrs(hexpr.base).concat([`${hexpr.axis}`]));
    }
}

export interface HSumr extends HTop {
    etype: HEType.Top;
    topType: HETopType.Sumr;
    base: HExpr;
    axis: number;
}

export namespace HSumr {
    export function create(base: HExpr, axis: number, source?: CodeSource): HSumr {
        return {
            etype: HEType.Top,
            topType: HETopType.Sumr,
            base,
            axis,
            source,
        };
    }

    export function toStrs(hexpr: HSumr): string[] {
        return HExpr.prefixStr('sumr', HExpr.toStrs(hexpr.base).concat([`${hexpr.axis}`]));
    }
}

export interface HConvol extends HTop {
    etype: HEType.Top;
    topType: HETopType.Convol;
    fmap: HExpr;
    kernel: HExpr;
}

export namespace HConvol {
    export function create(fmap: HExpr, kernel: HExpr, source?: CodeSource): HConvol {
        return {
            etype: HEType.Top,
            topType: HETopType.Convol,
            fmap,
            kernel,
            source,
        };
    }

    export function toStrs(hexpr: HConvol): string[] {
        return HExpr.prefixStr('convol', HExpr.toStrs(hexpr.fmap).concat(HExpr.toStrs(hexpr.kernel)));
    }
}

export interface HConvolp extends HTop {
    etype: HEType.Top;
    topType: HETopType.Convolp;
    fmap: HExpr;
    kernel: HExpr;
}

export namespace HConvolp {
    export function create(fmap: HExpr, kernel: HExpr, source?: CodeSource): HConvolp {
        return {
            etype: HEType.Top,
            topType: HETopType.Convolp,
            fmap,
            kernel,
            source,
        };
    }

    export function toStrs(hexpr: HConvolp): string[] {
        return HExpr.prefixStr('convolp', HExpr.toStrs(hexpr.fmap).concat(HExpr.toStrs(hexpr.kernel)));
    }
}

export interface HMaxpool extends HTop {
    etype: HEType.Top;
    topType: HETopType.Maxpool;
    fmap: HExpr;
    kh: number;
    kw: number;
}

export namespace HMaxpool {
    export function create(fmap: HExpr, kh: number, kw: number, source?: CodeSource): HMaxpool {
        return {
            etype: HEType.Top,
            topType: HETopType.Maxpool,
            fmap,
            kh,
            kw,
            source,
        };
    }

    export function toStrs(hexpr: HMaxpool): string[] {
        return HExpr.prefixStr('maxpool', HExpr.toStrs(hexpr.fmap).concat([`${hexpr.kh}`, `${hexpr.kw}`]));
    }
}

export interface HMaxpoolp extends HTop {
    etype: HEType.Top;
    topType: HETopType.Maxpoolp;
    fmap: HExpr;
    kh: number;
    kw: number;
}

export namespace HMaxpoolp {
    export function create(fmap: HExpr, kh: number, kw: number, source?: CodeSource): HMaxpoolp {
        return {
            etype: HEType.Top,
            topType: HETopType.Maxpoolp,
            fmap,
            kh,
            kw,
            source,
        };
    }

    export function toStrs(hexpr: HMaxpoolp): string[] {
        return HExpr.prefixStr('maxpoolp', HExpr.toStrs(hexpr.fmap).concat([`${hexpr.kh}`, `${hexpr.kw}`]));
    }
}

export interface HAveragepool extends HTop {
    etype: HEType.Top;
    topType: HETopType.Averagepool;
    fmap: HExpr;
    kh: number;
    kw: number;
}

export namespace HAveragepool {
    export function create(fmap: HExpr, kh: number, kw: number, source?: CodeSource): HAveragepool {
        return {
            etype: HEType.Top,
            topType: HETopType.Averagepool,
            fmap,
            kh,
            kw,
            source,
        };
    }

    export function toStrs(hexpr: HAveragepool): string[] {
        return HExpr.prefixStr('avgpool', HExpr.toStrs(hexpr.fmap).concat([`${hexpr.kh}`, `${hexpr.kw}`]));
    }
}

export interface HAveragepoolp extends HTop {
    etype: HEType.Top;
    topType: HETopType.Averagepoolp;
    fmap: HExpr;
    kh: number;
    kw: number;
}

export namespace HAveragepoolp {
    export function create(fmap: HExpr, kh: number, kw: number, source?: CodeSource): HAveragepoolp {
        return {
            etype: HEType.Top,
            topType: HETopType.Averagepoolp,
            fmap,
            kh,
            kw,
            source,
        };
    }

    export function toStrs(hexpr: HAveragepoolp): string[] {
        return HExpr.prefixStr('avgpoolp', HExpr.toStrs(hexpr.fmap).concat([`${hexpr.kh}`, `${hexpr.kw}`]));
    }
}

export interface HMap extends HTop {
    etype: HEType.Top;
    topType: HETopType.Map;
    uopType: HEUopType;
    base: HExpr;
}

export namespace HMap {
    export function create(uopType: HEUopType, base: HExpr, source?: CodeSource): HMap {
        return {
            etype: HEType.Top,
            topType: HETopType.Map,
            uopType,
            base,
            source,
        };
    }

    function uopChar(hmap: HMap): string {
        switch (hmap.uopType) {
            case HEUopType.Relu:
                return 'relu';
            case HEUopType.Sigmoid:
                return 'sigmoid';
            case HEUopType.Rtangent:
                return 'rtangent';
        }
    }

    export function toStrs(hexpr: HMap): string[] {
        return HExpr.prefixStr(uopChar(hexpr), HExpr.toStrs(hexpr.base));
    }
}
