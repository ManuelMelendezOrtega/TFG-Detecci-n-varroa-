package com.sistemasdistr.basico.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDate;
import java.util.List;

@Entity
@Data 
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "sesion_muestreo")
public class SesionMuestreo {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "usuario_id", nullable = false)
    private User usuario;

    @Column(nullable = false)
    private LocalDate fecha;

    @Column(name = "num_fotos", nullable = false)
    private Integer numFotos;

    @Column(name = "media_varroas", nullable = false)
    private Float mediaVarroas;

    @OneToMany(mappedBy = "sesion", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<FotoDetalle> fotos;
}