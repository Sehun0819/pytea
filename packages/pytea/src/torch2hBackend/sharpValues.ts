/*
 * sharpValues.ts
 * Copyright (c) Seoul National University.
 * Licensed under the MIT license.
 * Author: Ho Young Jhoo (mersshs@gmail.com)
 *
 * Values for PyTea internal languages with Immutable.js
 */
import { List, Map, Record } from 'immutable';

import { Range } from 'pyright-internal/common/textRange';
import { ParseNode } from 'pyright-internal/parser/parseNodes';

import { ThStmt, TSFunDef, TSPass } from '../frontend/torchStatements';
import { HExpr } from './hExpression';
import { ShEnv, ShHeap } from './sharpEnvironments';

export type ShValue =
    | SVAddr
    | SVInt
    | SVFloat
    | SVString
    | SVBool
    | SVObject
    | SVFunc
    | SVNone
    | SVNotImpl
    | SVUndef
    | SVTensor
    | SVError;

export type SVNumber = SVInt | SVFloat;
export type SVNumeric = SVInt | SVFloat | SVBool;
export type SVLiteral = SVInt | SVFloat | SVBool | SVString;

export const enum ShContFlag {
    Run,
    Cnt,
    Brk,
}

export const enum SVType {
    Addr,
    Int,
    Float,
    String,
    Bool,
    Object,
    Func,
    None,
    NotImpl,
    Undef,
    Tensor,
    Error,
}

export type SVNumberType = SVType.Int | SVType.Float;
export type SVNumericType = SVType.Int | SVType.Float | SVType.Bool;
export type SVLiteralType = SVType.Int | SVType.Float | SVType.Bool | SVType.String;

// defined in builtins.py
export enum PrimitiveType {
    Int = 0,
    Float = 1,
    Str = 2,
    Bool = 3,
    Tuple = 4,
    List = 5,
    Dict = 6,
    Set = 7,
    Ellipsis = 8,
}

let _svId = 0;
export function getNextSVId(): number {
    return ++_svId;
}

export type CodeSource = ParseNode | CodeRange;
export interface CodeRange {
    // file index (managed from PyteaService)
    fileId: number;

    // code range
    range: Range;
}

interface ShValueBase {
    readonly type: SVType;
    readonly source: CodeSource | undefined;
}

export namespace ShValue {
    export function toStringType(type: SVType | undefined): string {
        if (type === undefined) return 'undefined';

        switch (type) {
            case SVType.Addr:
                return 'Addr';
            case SVType.Int:
                return 'Int';
            case SVType.Float:
                return 'Float';
            case SVType.String:
                return 'String';
            case SVType.Bool:
                return 'Bool';
            case SVType.Object:
                return 'Object';
            case SVType.Func:
                return 'Func';
            case SVType.None:
                return 'None';
            case SVType.NotImpl:
                return 'NotImpl';
            case SVType.Undef:
                return 'Undef';
            case SVType.Tensor:
                return 'Tensor';
            case SVType.Error:
                return 'Error';
        }
    }

    export function toString(value: ShValue | ShContFlag): string {
        if (typeof value === 'object') {
            return value.toString();
        } else {
            switch (value) {
                case ShContFlag.Run:
                    return 'RUN';
                case ShContFlag.Cnt:
                    return 'CNT';
                case ShContFlag.Brk:
                    return 'BRK';
            }
        }
    }

    export function toStringStrMap(map: Map<string, ShValue>): string {
        if (map.count() === 0) {
            return '{}';
        }
        const keyArr = [...map.keys()];
        keyArr.sort();
        return `{ ${keyArr.map((i) => `${i} => ${map.get(i)?.toString()}`).join(', ')} }`;
    }

    export function toStringNumMap(map: Map<number, ShValue>): string {
        if (map.count() === 0) {
            return '{}';
        }
        const keyArr = [...map.keys()];
        keyArr.sort((a, b) => a - b);
        return `{ ${keyArr.map((i) => `${i} => ${map.get(i)?.toString()}`).join(', ')} }`;
    }

    // add address offset
    export function addOffset(value: ShValue, offset: number): ShValue {
        let newVal = value;

        switch (newVal.type) {
            case SVType.Addr:
                newVal = newVal.addOffset(offset);
                break;
            case SVType.Object:
                newVal = newVal.set(
                    'attrs',
                    newVal.attrs.mapEntries(([k, attr]) => [k, addOffset(attr, offset)])
                );
                newVal = newVal.set(
                    'indices',
                    newVal.indices.mapEntries(([k, attr]) => [k, addOffset(attr, offset)])
                );
                newVal = newVal.set(
                    'keyValues',
                    newVal.keyValues.mapEntries(([k, attr]) => [k, addOffset(attr, offset)])
                );
                break;
            case SVType.Func:
                SVFunc;
                newVal = newVal.set(
                    'defaults',
                    newVal.defaults.mapEntries(([k, v]) => [k, addOffset(v, offset)])
                );
                if (newVal.funcEnv) {
                    newVal = newVal.set('funcEnv', newVal.funcEnv.addOffset(offset));
                }
                break;
            default:
                break;
        }

        return newVal;
    }
}

interface SVAddrProps extends ShValueBase {
    readonly type: SVType.Addr;
    readonly addr: number;
}

const svAddrDefaults: SVAddrProps = {
    type: SVType.Addr,
    addr: -1,
    source: undefined,
};

export class SVAddr extends Record(svAddrDefaults) implements SVAddrProps {
    readonly type!: SVType.Addr;

    constructor(values?: Partial<SVAddrProps>) {
        values ? super(values) : super();
    }

    static create(addr: number, source: CodeSource | undefined): SVAddr {
        const value: SVAddr = new SVAddr({
            addr,
            source,
        });
        return value;
    }

    toString(): string {
        return `Loc(${this.addr})`;
    }

    addOffset(offset: number): SVAddr {
        return this.addr >= 0 ? this.set('addr', this.addr + offset) : this;
    }
}

interface SVIntProps extends ShValueBase {
    readonly type: SVType.Int;
    readonly value: number;
}

const svIntDefaults: SVIntProps = {
    type: SVType.Int,
    value: 0,
    source: undefined,
};

export class SVInt extends Record(svIntDefaults) implements SVIntProps {
    readonly type!: SVType.Int;

    constructor(values?: Partial<SVIntProps>) {
        values ? super(values) : super();
    }

    static create(intValue: number, source: CodeSource | undefined): SVInt {
        const value: SVInt = new SVInt({
            value: intValue,
            source,
        });
        return value;
    }

    toString(): string {
        return this.value.toString();
    }
}

interface SVFloatProps extends ShValueBase {
    readonly type: SVType.Float;
    readonly value: number;
}

const svFloatDefaults: SVFloatProps = {
    type: SVType.Float,
    value: 0,
    source: undefined,
};

export class SVFloat extends Record(svFloatDefaults) implements SVFloatProps {
    readonly type!: SVType.Float;

    constructor(values?: Partial<SVFloatProps>) {
        values ? super(values) : super();
    }

    static create(floatValue: number, source: CodeSource | undefined): SVFloat {
        const value: SVFloat = new SVFloat({
            value: floatValue,
            source,
        });
        return value;
    }

    toString(): string {
        if (Number.isInteger(this.value)) {
            return `${this.value}.0`;
        } else {
            return this.value.toString();
        }
    }
}
interface SVStringProps extends ShValueBase {
    readonly type: SVType.String;
    readonly value: string;
}

const svStringDefaults: SVStringProps = {
    type: SVType.String,
    value: '',
    source: undefined,
};

export class SVString extends Record(svStringDefaults) implements SVStringProps {
    readonly type!: SVType.String;

    constructor(values?: Partial<SVStringProps>) {
        values ? super(values) : super();
    }

    static create(strValue: string, source: CodeSource | undefined): SVString {
        const value: SVString = new SVString({
            value: strValue,
            source,
        });
        return value;
    }

    toString(): string {
        return `"${this.value}"`;
    }
}

interface SVBoolProps extends ShValueBase {
    readonly type: SVType.Bool;
    readonly value: boolean;
}

const svBoolDefaults: SVBoolProps = {
    type: SVType.Bool,
    value: false,
    source: undefined,
};

export class SVBool extends Record(svBoolDefaults) implements SVBoolProps {
    readonly type!: SVType.Bool;

    constructor(values?: Partial<SVBoolProps>) {
        values ? super(values) : super();
    }

    static create(boolValue: boolean, source: CodeSource | undefined): SVBool {
        const value: SVBool = new SVBool({
            value: boolValue,
            source,
        });
        return value;
    }

    toString(): string {
        return this.value ? 'true' : 'false';
    }
}

interface SVObjectProps extends ShValueBase {
    readonly type: SVType.Object;
    readonly id: number;
    readonly attrs: Map<string, ShValue>;
    readonly indices: Map<number, ShValue>;
    readonly keyValues: Map<string, ShValue>;
    readonly addr: SVAddr;
}

const svObjectProps: SVObjectProps = {
    type: SVType.Object,
    id: -1,
    attrs: Map(), // TODO: default methods
    indices: Map(),
    keyValues: Map(),
    addr: SVAddr.create(Number.NEGATIVE_INFINITY, undefined),
    source: undefined,
};

export class SVObject extends Record(svObjectProps) implements SVObjectProps {
    readonly type!: SVType.Object;

    constructor(values?: Partial<SVObjectProps>) {
        values ? super(values) : super();
    }

    // from now on, object creation should be bind with address.
    static create(heap: ShHeap, source: CodeSource | undefined): [SVObject, SVAddr, ShHeap] {
        const [addr, newHeap] = heap.malloc(source);
        const value: SVObject = new SVObject({
            id: getNextSVId(),
            addr,
            source,
        });

        return [value, addr, newHeap.setVal(addr, value)];
    }

    // if address is fixed, use it and set addr after.
    static createWithAddr(addr: SVAddr, source: CodeSource | undefined): SVObject {
        const value: SVObject = new SVObject({
            id: getNextSVId(),
            addr,
            source,
        });

        return value;
    }

    setAttr(attr: string, value: ShValue): SVObject {
        return this.set('attrs', this.attrs.set(attr, value));
    }

    setIndice(index: number, value: ShValue): SVObject {
        return this.set('indices', this.indices.set(index, value));
    }

    setKeyVal(key: string, value: ShValue): SVObject {
        return this.set('keyValues', this.keyValues.set(key, value));
    }

    getAttr(attr: string): ShValue | undefined {
        return this.attrs.get(attr);
    }

    getIndice(index: number): ShValue | undefined {
        return this.indices.get(index);
    }

    getKeyVal(key: string): ShValue | undefined {
        return this.keyValues.get(key);
    }

    toString(): string {
        const attrStr = `${ShValue.toStringStrMap(this.attrs)}`;
        const indStr = `${ShValue.toStringNumMap(this.indices)}`;
        const kvStr = `${ShValue.toStringStrMap(this.keyValues)}`;
        return `[${this.addr.addr}]{ ${attrStr}, ${indStr}, ${kvStr} }`;
    }

    clone(heap: ShHeap, source: CodeSource | undefined): [SVObject, ShHeap] {
        const [addr, newHeap] = heap.malloc(source);
        const obj = this.set('addr', addr).set('id', getNextSVId()).set('source', source);
        return [obj, newHeap.setVal(addr, obj)];
    }
}

interface SVFuncProps extends ShValueBase {
    readonly type: SVType.Func;
    readonly id: number;
    readonly name: string;
    readonly params: List<string>;
    readonly defaults: Map<string, ShValue>;
    readonly funcBody: ThStmt;
    readonly hasClosure: boolean;
    readonly funcEnv?: ShEnv; // make it optional to avoid TypeScript circular import dependency
    readonly varargsParam?: string;
    readonly kwargsParam?: string;
    readonly keyOnlyNum?: number; // length of keyword-only argument. (with respect to varargs and PEP3012 keyword only arg)
}

const svFuncDefaults: SVFuncProps = {
    type: SVType.Func,
    id: -1,
    name: '',
    params: List(),
    defaults: Map(),
    funcBody: TSPass.get(undefined),
    hasClosure: false,
    funcEnv: undefined,
    varargsParam: undefined,
    kwargsParam: undefined,
    source: undefined,
    keyOnlyNum: undefined,
};

export class SVFunc extends Record(svFuncDefaults) implements SVFuncProps {
    readonly type!: SVType.Func;

    constructor(values?: Partial<SVFuncProps>) {
        values ? super(values) : super();
    }

    static create(
        name: string,
        params: List<string>,
        funcBody: ThStmt,
        funcEnv: ShEnv,
        source: CodeSource | undefined
    ): SVFunc {
        const value: SVFunc = new SVFunc({
            id: getNextSVId(),
            name,
            params,
            funcBody,
            funcEnv,
            hasClosure: TSFunDef.findClosure(funcBody),
            source,
        });
        return value;
    }

    setDefaults(defaults: Map<string, ShValue>): SVFunc {
        return this.set('defaults', defaults);
    }

    setVKParam(varargsParam?: string, kwargsParam?: string, keyOnlyNum?: number): SVFunc {
        let func: SVFunc = this;
        if (varargsParam !== undefined) {
            func = func.set('varargsParam', varargsParam);
        }
        if (kwargsParam !== undefined) {
            func = func.set('kwargsParam', kwargsParam);
        }
        if (keyOnlyNum !== undefined) {
            func = func.set('keyOnlyNum', keyOnlyNum);
        }

        return func;
    }

    toString(): string {
        return `${this.name}(${this.params.join(', ')})`;
    }

    bound(selfAddr: SVAddr): SVFunc {
        // TODO: staticmethod.
        // self value should be given as address.
        if (this.params.count() === 0) {
            return this;
        }
        const selfName = this.params.get(0)!;
        const boundParams = this.params.slice(1);
        const newEnv = this.funcEnv?.setId(selfName, selfAddr);

        return this.set('params', boundParams).set('funcEnv', newEnv);
    }
}

interface SVNoneProps extends ShValueBase {
    readonly type: SVType.None;
}

const svNoneDefaults: SVNoneProps = {
    type: SVType.None,
    source: undefined,
};

export class SVNone extends Record(svNoneDefaults) implements SVNoneProps {
    readonly type!: SVType.None;
    static _none = new SVNone();

    private constructor(values?: Partial<SVNoneProps>) {
        values ? super(values) : super();
    }

    static create(source?: CodeSource): SVNone {
        if (!source) return SVNone._none;
        const value: SVNone = new SVNone({
            source,
        });
        return value;
    }

    toString(): string {
        return 'None';
    }
}

interface SVNotImplProps extends ShValueBase {
    readonly type: SVType.NotImpl;
    readonly reason?: string;
}

const svNotImplDefaults: SVNotImplProps = {
    type: SVType.NotImpl,
    reason: undefined,
    source: undefined,
};

export class SVNotImpl extends Record(svNotImplDefaults) implements SVNotImplProps {
    readonly type!: SVType.NotImpl;
    static _notImpl = new SVNotImpl();

    private constructor(values?: Partial<SVNotImplProps>) {
        values ? super(values) : super();
    }

    static create(reason: string, source: CodeSource | undefined): SVNotImpl {
        if (!reason && !source) {
            return this._notImpl;
        }
        const value: SVNotImpl = new SVNotImpl({
            reason,
            source,
        });
        return value;
    }

    toString(): string {
        return `NotImpl(${this.reason ? this.reason : ''})`;
    }
}

interface SVUndefProps extends ShValueBase {
    readonly type: SVType.Undef;
}

const svUndefDefaults: SVUndefProps = {
    type: SVType.Undef,
    source: undefined,
};

export class SVUndef extends Record(svUndefDefaults) implements SVUndefProps {
    readonly type!: SVType.Undef;
    static _undef = new SVUndef();

    private constructor(values?: Partial<SVUndefProps>) {
        values ? super(values) : super();
    }

    static create(source: CodeSource | undefined): SVUndef {
        if (!source) {
            return SVUndef._undef;
        }
        const value: SVUndef = new SVUndef({
            source,
        });
        return value;
    }

    toString(): string {
        return 'UNDEF';
    }
}

interface SVTensorProps extends ShValueBase {
    readonly type: SVType.Tensor;
    readonly hexpr: HExpr | undefined;
}

const svTensorDefaults: SVTensorProps = {
    type: SVType.Tensor,
    hexpr: undefined,
    source: undefined,
};

export class SVTensor extends Record(svTensorDefaults) implements SVTensorProps {
    readonly type!: SVType.Tensor;

    constructor(hexpr?: Partial<SVTensorProps>) {
        hexpr ? super(hexpr) : super();
    }

    static create(hexpr: HExpr, source: CodeSource | undefined): SVTensor {
        const value: SVTensor = new SVTensor({
            hexpr,
            source,
        });
        return value;
    }

    toString(): string {
        return this.hexpr === undefined ? 'undefined' : HExpr.toString(this.hexpr);
    }
}

export interface SVErrorProps extends ShValueBase {
    readonly type: SVType.Error;
    readonly reason: string;
    readonly level: SVErrorLevel;
}

const svErrorDefaults: SVErrorProps = {
    type: SVType.Error,
    reason: 'unexpected error',
    level: SVErrorLevel.Error,
    source: undefined,
};

export const enum SVErrorLevel {
    Error,
    Warning,
    Log,
}
export class SVError extends Record(svErrorDefaults) implements SVErrorProps {
    readonly type!: SVType.Error;

    private constructor(values?: Partial<SVErrorProps>) {
        values ? super(values) : super();
    }

    static create(reason: string, level: SVErrorLevel, source: CodeSource | undefined): SVError {
        const value: SVError = new SVError({
            reason,
            level,
            source,
        });
        return value;
    }

    static error(reason: string, source: CodeSource | undefined): SVError {
        return this.create(reason, SVErrorLevel.Error, source);
    }

    static warn(reason: string, source: CodeSource | undefined): SVError {
        return this.create(reason, SVErrorLevel.Warning, source);
    }

    static log(reason: string, source: CodeSource | undefined): SVError {
        return this.create(reason, SVErrorLevel.Log, source);
    }

    toString(): string {
        // const pos = formatParseNode(this.source);
        return `SVError<${svErrorLevelToString(this.level)}: "${this.reason}">`;
    }
}

export function svErrorLevelToString(level: SVErrorLevel): string {
    switch (level) {
        case SVErrorLevel.Error:
            return 'Error';
        case SVErrorLevel.Warning:
            return 'Warning';
        case SVErrorLevel.Log:
            return 'Log';
    }
}

export function svTypeToString(type: SVType) {
    switch (type) {
        case SVType.Addr:
            return 'Addr';
        case SVType.Int:
            return 'Int';
        case SVType.Float:
            return 'Float';
        case SVType.String:
            return 'String';
        case SVType.Bool:
            return 'Bool';
        case SVType.Object:
            return 'Object';
        case SVType.Func:
            return 'Func';
        case SVType.None:
            return 'None';
        case SVType.NotImpl:
            return 'NotImpl';
        case SVType.Undef:
            return 'Undef';
        case SVType.Error:
            return 'Error';
    }
}
