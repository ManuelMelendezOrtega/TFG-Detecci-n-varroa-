package com.tfg.basico.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "foto_detalle")
public class FotoDetalle {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "sesion_id", nullable = false)
    private SesionMuestreo sesion;

    @Column(name = "conteo_varroas", nullable = false)
    private Integer conteoVarroas;

    @Column(name = "ruta_imagen_original")
    private String rutaImagenOriginal;

    @Column(name = "ruta_imagen_marcada")
    private String rutaImagenMarcada;
}